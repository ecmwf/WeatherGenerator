# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Optimizer module for WeatherGenerator.

Provides support for:
- Standard AdamW optimizer
- Hybrid Muon+AdamW optimizer (Muon for 2D hidden weights, AdamW for embeddings/heads)

The Muon optimizer uses orthogonalization of gradients for improved training dynamics
on transformer hidden layer weights. See: https://arxiv.org/abs/2407.01490
"""

import logging
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# Patterns identifying parameters that should use AdamW (not Muon)
# These include embeddings, prediction heads, and other 1D or special parameters
ADAMW_PATTERNS = [
    "embed_target_coords",
    "embeds.",
    "embed.",
    "unembed",
    "pred_heads",
    "latent_heads",
    "q_cells",
    "bilin",
    "class_token",
    "register_token",
    "norm",
    "bias",
]


def classify_muon_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[str], list[str]]:
    """
    Classify model parameters into Muon-eligible and AdamW-eligible groups.

    Muon is applied to 2D hidden layer weights (attention Q/K/V/O, MLP linear layers).
    AdamW is applied to embeddings, output heads, 1D parameters, and biases.

    Args:
        model: The model whose parameters to classify.

    Returns:
        A tuple of (muon_params, adamw_params, muon_names, adamw_names).
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # 1D parameters (biases, layer norm weights) -> AdamW
        if param.ndim < 2:
            adamw_params.append(param)
            adamw_names.append(name)
            continue

        # Check if parameter matches any AdamW pattern
        is_adamw = any(pattern in name_lower for pattern in ADAMW_PATTERNS)

        if is_adamw:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            # 2D hidden weights -> Muon
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


def _scale_adamw_betas(
    beta1_base: float,
    beta2_base: float,
    eps_base: float,
    batch_size_total: int,
) -> tuple[float, float, float]:
    """
    Scale AdamW hyperparameters based on batch size following SDE scaling rules.

    See: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/

    Args:
        beta1_base: Base beta1 value (target for batch_size_total=1).
        beta2_base: Base beta2 value (target for batch_size_total=1).
        eps_base: Base epsilon value.
        batch_size_total: Total effective batch size across all ranks.

    Returns:
        Tuple of (scaled_beta1, scaled_beta2, scaled_eps).
    """
    kappa = batch_size_total
    beta1 = max(0.5, 1.0 - kappa * (1.0 - beta1_base))
    beta2 = 1.0 - kappa * (1.0 - beta2_base)
    eps = eps_base / np.sqrt(kappa)
    return beta1, beta2, eps


def create_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: Any,
    lr_cfg: Any,
    batch_size_total: int,
) -> Optimizer:
    """
    Factory function to create the appropriate optimizer based on config.

    Args:
        model: The model to optimize.
        optimizer_cfg: Optimizer configuration containing type and hyperparameters.
        lr_cfg: Learning rate configuration containing lr_start.
        batch_size_total: Total effective batch size across all ranks.

    Returns:
        The configured optimizer (AdamW or CompositeOptimizer).
    """
    optimizer_type = optimizer_cfg.get("type", "adamw")
    initial_lr = lr_cfg.lr_start
    weight_decay = optimizer_cfg.weight_decay

    # Scale AdamW betas based on batch size
    adamw_cfg = optimizer_cfg.adamw
    beta1, beta2, eps = _scale_adamw_betas(
        adamw_cfg.beta1,
        adamw_cfg.beta2,
        adamw_cfg.get("eps", 2e-08),
        batch_size_total,
    )

    if optimizer_type == "adamw":
        logger.info("Creating AdamW optimizer")
        return torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )

    elif optimizer_type == "muon_adamw":
        logger.info("Creating Muon+AdamW composite optimizer")
        return _create_muon_adamw_optimizer(
            model=model,
            optimizer_cfg=optimizer_cfg,
            initial_lr=initial_lr,
            weight_decay=weight_decay,
            adamw_betas=(beta1, beta2),
            adamw_eps=eps,
            batch_size_total=batch_size_total,
        )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def _create_muon_adamw_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: Any,
    initial_lr: float,
    weight_decay: float,
    adamw_betas: tuple[float, float],
    adamw_eps: float,
    batch_size_total: int,
) -> "CompositeOptimizer":
    """
    Create a Muon+AdamW composite optimizer.

    Args:
        model: The model to optimize.
        optimizer_cfg: Optimizer configuration.
        initial_lr: Initial learning rate (for AdamW; Muon uses multiplied version).
        weight_decay: Weight decay coefficient.
        adamw_betas: Scaled (beta1, beta2) for AdamW.
        adamw_eps: Scaled epsilon for AdamW.
        batch_size_total: Total effective batch size.

    Returns:
        CompositeOptimizer wrapping Muon and AdamW.
    """
    muon_cfg = optimizer_cfg.get("muon", {})
    lr_multiplier = muon_cfg.get("lr_multiplier", 20.0)
    muon_momentum = muon_cfg.get("momentum", 0.95)
    muon_nesterov = muon_cfg.get("nesterov", True)
    muon_weight_decay = muon_cfg.get("weight_decay", weight_decay)

    # Classify parameters
    muon_params, adamw_params, muon_names, adamw_names = classify_muon_params(model)

    logger.info(f"Muon parameters ({len(muon_params)}): {muon_names[:5]}...")
    logger.info(f"AdamW parameters ({len(adamw_params)}): {adamw_names[:5]}...")

    # Create parameter groups for AdamW
    # Include both AdamW-only params and mark them appropriately
    adamw_param_groups = [
        {
            "params": adamw_params,
            "lr": initial_lr,
            "is_muon": False,
            "lr_multiplier": 1.0,
        }
    ]

    # Create AdamW optimizer for embeddings/heads
    adamw_optimizer = torch.optim.AdamW(
        adamw_param_groups,
        lr=initial_lr,
        weight_decay=weight_decay,
        betas=adamw_betas,
        eps=adamw_eps,
    )

    # Create Muon optimizer for hidden weights
    muon_lr = initial_lr * lr_multiplier

    # Parameter groups for Muon
    muon_param_groups = [
        {
            "params": muon_params,
            "lr": muon_lr,
            "is_muon": True,
            "lr_multiplier": lr_multiplier,
        }
    ]

    # Try to use PyTorch's built-in Muon if available (PyTorch >= 2.9)
    muon_optimizer = _create_muon_optimizer(
        param_groups=muon_param_groups,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        weight_decay=muon_weight_decay,
    )

    return CompositeOptimizer(
        muon_optimizer=muon_optimizer,
        adamw_optimizer=adamw_optimizer,
        muon_lr_multiplier=lr_multiplier,
    )


def _create_muon_optimizer(
    param_groups: list[dict],
    lr: float,
    momentum: float,
    nesterov: bool,
    weight_decay: float,
) -> Optimizer:
    """
    Create a Muon optimizer, using PyTorch's built-in version if available.

    Falls back to custom implementation for older PyTorch versions.
    """
    # Try PyTorch's built-in Muon (available in PyTorch >= 2.9)
    if hasattr(torch.optim, "Muon"):
        logger.info("Using torch.optim.Muon")
        return torch.optim.Muon(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    else:
        logger.info("Using custom Muon implementation (torch.optim.Muon not available)")
        return MuonCustom(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )


class CompositeOptimizer(Optimizer):
    """
    Composite optimizer that combines Muon and AdamW for different parameter groups.

    Muon is used for 2D hidden layer weights, AdamW for embeddings and heads.
    This class wraps both optimizers and provides a unified interface.

    Inherits from Optimizer for compatibility with PyTorch LR schedulers.
    """

    def __init__(
        self,
        muon_optimizer: Optimizer,
        adamw_optimizer: Optimizer,
        muon_lr_multiplier: float = 20.0,
    ):
        """
        Initialize the composite optimizer.

        Args:
            muon_optimizer: Optimizer for Muon-eligible parameters.
            adamw_optimizer: Optimizer for AdamW-eligible parameters.
            muon_lr_multiplier: LR multiplier for Muon relative to base LR.
        """
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer
        self.muon_lr_multiplier = muon_lr_multiplier

        # Manually initialize Optimizer base class attributes without calling __init__
        # This avoids the param_groups setup that would conflict with our combined groups
        from collections import OrderedDict, defaultdict

        # Set defaults with betas for LR scheduler compatibility (OneCycleLR checks this)
        # Use AdamW's betas since that's the more common scheduler interaction
        adamw_betas = adamw_optimizer.defaults.get("betas", (0.9, 0.999))
        self.defaults = {
            "betas": adamw_betas,
            "momentum": muon_optimizer.defaults.get("momentum", 0.95),
        }
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

        # Ensure all param groups have betas for OneCycleLR compatibility
        # OneCycleLR with cycle_momentum=True tries to modify betas on ALL groups
        for group in muon_optimizer.param_groups:
            if "betas" not in group:
                group["betas"] = adamw_betas

        # Combined param_groups from both optimizers
        self.param_groups = muon_optimizer.param_groups + adamw_optimizer.param_groups

        # State is a combined view (we override the property below)
        self._state = defaultdict(dict)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional for most optimizers.

        Returns:
            Loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.muon_optimizer.step()
        self.adamw_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """
        Reset gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
                This can improve memory efficiency.
        """
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """
        Return the state of both optimizers as a single dictionary.

        Returns:
            Dictionary containing state from both Muon and AdamW optimizers.
        """
        return {
            "muon": self.muon_optimizer.state_dict(),
            "adamw": self.adamw_optimizer.state_dict(),
            "muon_lr_multiplier": self.muon_lr_multiplier,
            "optimizer_type": "composite_muon_adamw",
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load optimizer state from a dictionary.

        Args:
            state_dict: Dictionary containing saved optimizer state.
        """
        if (
            "optimizer_type" in state_dict
            and state_dict["optimizer_type"] == "composite_muon_adamw"
        ):
            self.muon_optimizer.load_state_dict(state_dict["muon"])
            self.adamw_optimizer.load_state_dict(state_dict["adamw"])
            self.muon_lr_multiplier = state_dict.get("muon_lr_multiplier", self.muon_lr_multiplier)
        else:
            # Fallback: try to load as regular optimizer state
            # This handles migration from pure AdamW checkpoints
            logger.warning(
                "Loading non-composite state dict into CompositeOptimizer. "
                "This may not work correctly - optimizer state may be lost."
            )

    @property
    def state(self) -> dict:
        """
        Return combined state from both optimizers.

        This provides a unified view of optimizer state for checkpointing.
        """
        combined_state = dict(self._state)
        combined_state.update(self.muon_optimizer.state)
        combined_state.update(self.adamw_optimizer.state)
        return combined_state

    @state.setter
    def state(self, value):
        """Set state (needed for Optimizer base class compatibility)."""
        self._state = value


class MuonCustom(Optimizer):
    """
    Custom Muon optimizer implementation for PyTorch versions without torch.optim.Muon.

    Muon applies Newton-Schulz orthogonalization to gradients before the SGD update,
    which helps with optimization of transformer hidden layer weights.

    Reference: https://arxiv.org/abs/2407.01490
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        """
        Initialize the Muon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining param groups.
            lr: Learning rate.
            momentum: Momentum factor.
            nesterov: Whether to use Nesterov momentum.
            weight_decay: Weight decay (L2 penalty).
            ns_steps: Number of Newton-Schulz iterations for orthogonalization.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            ns_steps = group.get("ns_steps", 5)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Apply Newton-Schulz orthogonalization for 2D+ tensors
                if p.ndim >= 2:
                    grad = self._newton_schulz_orthogonalize(grad, ns_steps)

                # Get or initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

                # Apply update
                p.add_(grad, alpha=-lr)

        return loss

    def _newton_schulz_orthogonalize(self, grad: torch.Tensor, ns_steps: int) -> torch.Tensor:
        """
        Apply Newton-Schulz iteration to orthogonalize the gradient.

        This projects the gradient onto the manifold of orthogonal matrices,
        which helps with optimization stability for large matrices.

        Args:
            grad: Gradient tensor to orthogonalize.
            ns_steps: Number of Newton-Schulz iterations.

        Returns:
            Orthogonalized gradient tensor.
        """
        # Reshape to 2D if needed
        original_shape = grad.shape
        if grad.ndim > 2:
            grad = grad.view(grad.shape[0], -1)

        # Transpose if needed to ensure we have more rows than columns
        transposed = False
        if grad.shape[0] < grad.shape[1]:
            grad = grad.T
            transposed = True

        # Normalize
        grad = grad / (grad.norm() + 1e-7)

        # Newton-Schulz iteration: X_{k+1} = X_k (3I - X_k^T X_k) / 2
        # This converges to an orthogonal matrix
        for _ in range(ns_steps):
            grad = grad @ (
                1.5 * torch.eye(grad.shape[1], device=grad.device, dtype=grad.dtype)
                - 0.5 * grad.T @ grad
            )

        # Restore original orientation
        if transposed:
            grad = grad.T

        # Reshape back to original
        grad = grad.view(original_shape)

        return grad
