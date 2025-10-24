# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from weathergen.model.norms import AdaLayerNorm, RMSNorm


class NamedLinear(torch.nn.Module):
    def __init__(self, name: str | None = None, **kwargs):
        super(NamedLinear, self).__init__()
        self.linear = nn.Linear(**kwargs)
        if name is not None:
            self.name = name

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_layers=2,
        hidden_factor=2,
        pre_layer_norm=True,
        dropout_rate=0.0,
        nonlin=torch.nn.GELU,
        with_residual=False,
        norm_type="LayerNorm",
        dim_aux=None,
        norm_eps=1e-5,
        name: str | None = None,
    ):
        """Constructor"""

        super(MLP, self).__init__()

        if name is not None:
            self.name = name

        assert num_layers >= 2

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        dim_hidden = int(dim_in * hidden_factor)

        self.layers = torch.nn.ModuleList()

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        if pre_layer_norm:
            self.layers.append(
                norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )

        self.layers.append(torch.nn.Linear(dim_in, dim_hidden))
        self.layers.append(nonlin())
        self.layers.append(torch.nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nonlin())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

        self.layers.append(torch.nn.Linear(dim_hidden, dim_out))

    def forward(self, *args):
        x, x_in, aux = args[0], args[0], args[-1]

        for i, layer in enumerate(self.layers):
            x = layer(x, aux) if (i == 0 and self.with_aux) else layer(x)

        if self.with_residual:
            if x.shape[-1] == x_in.shape[-1]:
                x = x_in + x
            else:
                assert x.shape[-1] % x_in.shape[-1] == 0
                x = x + x_in.repeat([*[1 for _ in x.shape[:-1]], x.shape[-1] // x_in.shape[-1]])

        return x

class _DenseBlock(nn.Module):
    """A tiny FFN that mirrors the structure of the current MLP stack."""
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers=2,
                 nonlin=nn.GELU, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_hidden), nonlin(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(dim_hidden, dim_hidden), nonlin(), nn.Dropout(dropout_rate)]
        layers += [nn.Linear(dim_hidden, dim_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MoEMLP(nn.Module):
    """
    Memory-friendly MoE MLP.

    Features
    --------
    - Matches MLP call pattern: forward(*args) where args=(x, ...) and optional aux at the end
    - Optional AdaLayerNorm pre-norm when dim_aux is provided
    - Top-k routing with softmax over selected logits
    - Streams experts and accumulates outputs (no large [E, ..., D] stacks)
    - Optional auxiliary outputs (gate loss, route histogram) via `return_aux`

    Notes
    -----
    - If `return_aux=False` (default), we still *compute* the aux loss (with grads) and stash it
      on `self.last_aux` and `self.last_aux_loss` so you can read it after forward if desired.
    - To actively use the load-balancing loss in training, either set `return_aux=True` and add it
      to your loss, or read `self.last_aux['gate_loss']` from the module instance.
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_layers: int = 2,
        hidden_factor: float = 2.0,
        pre_layer_norm: bool = True,
        dropout_rate: float = 0.0,
        nonlin=nn.GELU,
        with_residual: bool = False,
        norm_type: str = "LayerNorm",
        dim_aux: Optional[int] = None,
        norm_eps: float = 1e-5,
        name: Optional[str] = None,
        # MoE
        num_experts: int = 8,
        top_k: int = 4,
        router_noisy_std: float = 0.0,
        # Memory
        use_checkpoint: bool = False,
        # API
        return_aux: bool = False,
    ):
        super().__init__()
        if name is not None:
            self.name = name

        assert num_layers >= 2, "MoEMLP requires at least 2 layers"
        assert 1 <= top_k <= num_experts, "top_k must be in [1, num_experts]"

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        self.pre_layer_norm = pre_layer_norm
        self.top_k = top_k
        self.num_experts = num_experts
        self.router_noisy_std = router_noisy_std
        self.use_checkpoint = use_checkpoint
        self.return_aux = return_aux
        self.enable_gate_loss = True

        self.register_buffer("usage_buf", torch.zeros(num_experts), persistent=False)
        dim_hidden = int(dim_in * hidden_factor)

        # Norm (match MLP behavior)
        Norm = nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm
        if pre_layer_norm:
            self.norm = (
                Norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )
        else:
            self.norm = None

        # Router
        self.router = nn.Linear(dim_in, num_experts)
        # Recommended init: small std, zero bias
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.router.bias, 0.0)

        # Experts
        self.experts = nn.ModuleList(
            [
                _DenseBlock(
                    dim_in=dim_in,
                    dim_hidden=dim_hidden,
                    dim_out=dim_out,
                    num_layers=num_layers,
                    nonlin=nonlin,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_experts)
            ]
        )

        # Stashed aux for consumers that don't use return_aux
        self.register_buffer("last_aux_loss", torch.zeros((), dtype=torch.float32))
        self.last_aux: Dict[str, torch.Tensor] = {}

    def _gate(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            weights: [..., E] if top_k == E else [..., K]
            top_idx: None if full softmax, else [..., K] int indices
        """
        logits = self.router(x_norm)
        if self.router_noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noisy_std

        if self.top_k == self.num_experts:
            weights = torch.softmax(logits, dim=-1)
            top_idx = None
        else:
            top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
            weights = torch.softmax(top_vals, dim=-1)
        return weights, top_idx

    def _compute_load_balance_aux(
        self, weights: torch.Tensor, top_idx: Optional[torch.Tensor], num_experts: int
    ) -> torch.Tensor:
        """
        Cross-entropy between observed expert usage and uniform 1/E target.
        Works for both full-softmax and top-k.
        """
        if top_idx is None:
            # weights over E -> average across batch/time dims
            probs = weights.mean(dim=tuple(range(weights.dim() - 1)))  # [E]
        else:
            # Aggregate usage from top-k selections
            if weights.shape != top_idx.shape:
                raise ValueError("Top-k weights and indices must share the same shape")
            K = weights.shape[-1]
            flat_w = weights.reshape(-1, K)  # [N, K]
            flat_i = top_idx.reshape(-1, K)  # [N, K]
            usage = torch.zeros(num_experts, device=weights.device, dtype=weights.dtype)
            usage.scatter_add_(0, flat_i.reshape(-1), flat_w.reshape(-1))
            probs = usage / usage.sum().clamp_min(1e-6)  # [E]

        E = num_experts
        target = torch.full_like(probs, 1.0 / E)
        aux = (probs * (probs.add(1e-6).log() - target.add(1e-6).log())).sum()
        return aux

    def forward(self, *args):
        """
        Args:
            *args: expects x first; if AdaLN is enabled (dim_aux != None), the last arg is aux.

        Returns:
            y or (y, aux_out) depending on `self.return_aux`.
            aux_out = {"gate_loss": ..., "route_hist": ...}
        """
        x = args[0]
        x_in = x
        aux_in = args[-1] if self.with_aux else None

        # Optional pre-norm
        if self.norm is not None:
            x = self.norm(x, aux_in) if self.with_aux else self.norm(x)

        # Routing
        weights, top_idx = self._gate(x)  # [..., E] or [..., K]

        # Build full weights when in top-k mode to stream experts
        if top_idx is None:
            w_full = weights  # [..., E]
        else:
            E = self.num_experts
            w_full = torch.zeros(*weights.shape[:-1], E, device=weights.device, dtype=weights.dtype)
            w_full.scatter_(-1, top_idx, weights)

        # Accumulate outputs without stacking
        out_dim = self.experts[0].net[-1].out_features  # last Linear of _DenseBlock
        y = x.new_zeros(*x.shape[:-1], out_dim)

        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint

        for e, expert in enumerate(self.experts):
            w_e = w_full[..., e]  # [...]
            # Skip experts with (near) zero mass
            if w_e.abs().max() <= 1e-12:
                continue
            y_e = expert(x) if not (self.use_checkpoint and self.training) else checkpoint(expert, x)
            y = y + y_e * w_e.unsqueeze(-1)

        # Residual
        if self.with_residual:
            if y.shape[-1] == x_in.shape[-1]:
                y = x_in + y
            else:
                assert y.shape[-1] % x_in.shape[-1] == 0
                y = y + x_in.repeat([*[1 for _ in y.shape[:-1]], y.shape[-1] // x_in.shape[-1]])

        # # Aux outputs (WITH grads so router learns; also stash for external access)
        # aux_out: Dict[str, Any] = {}
        # gate_loss = self._compute_load_balance_aux(weights, top_idx, self.num_experts)
        # aux_out["gate_loss"] = gate_loss

        # # utilization histogram (for logging)
        # if top_idx is None:
        #     aux_out["route_hist"] = weights.mean(dim=tuple(range(weights.dim() - 1)))  # [E]
        # else:
        #     K = weights.shape[-1]
        #     flat_w = weights.reshape(-1, K)
        #     flat_i = top_idx.reshape(-1, K)
        #     usage = torch.zeros(self.num_experts, device=weights.device, dtype=weights.dtype)
        #     usage.scatter_add_(0, flat_i.reshape(-1), flat_w.reshape(-1))
        #     aux_out["route_hist"] = usage / usage.sum().clamp_min(1e-6)  # [E]

        # # stash for consumers that don't use return_aux
        # self.last_aux = aux_out
        # self.last_aux_loss = gate_loss

        # return (y, aux_out) if self.return_aux else y
        # --- Aux outputs (gate loss + route hist) ---
        aux_out: Dict[str, Any] = {}
        if self.enable_gate_loss:
            gate_loss = self._compute_load_balance_aux(weights, top_idx, self.num_experts)
            aux_out["gate_loss"] = gate_loss

            # utilization histogram (for logging / debugging only)
            if top_idx is None:
                aux_out["route_hist"] = weights.mean(dim=tuple(range(weights.dim() - 1)))  # [E]
            else:
                K = weights.shape[-1]
                flat_w = weights.reshape(-1, K)
                flat_i = top_idx.reshape(-1, K)
                usage = self.usage_buf
                usage = usage.to(weights.device, dtype=weights.dtype)
                usage.zero_()
                usage.scatter_add_(0, flat_i.reshape(-1), flat_w.reshape(-1))
                aux_out["route_hist"] = usage / usage.sum().clamp_min(1e-6)  # [E]
        else:
            # no aux computation this step
            pass

        # stash
        self.last_aux = aux_out
        if "gate_loss" in aux_out:
            self.last_aux_loss = aux_out["gate_loss"]

        return (y, aux_out) if self.return_aux else y