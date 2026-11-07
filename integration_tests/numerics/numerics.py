# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Forward/backward numerics regression harness for the training techniques
(forecasting, diffusion, spatial JEPA, temporal JEPA), see issue #2948.

One training step (forward, target computation, loss, backward) of a tiny model is run on a
single, fixed ERA5 batch. On the baseline code version the `generate` command stores a golden
reference per technique:

    <ref_dir>/<technique>/inputs.pt   config, batch, trainable student weights, target encoder
    <ref_dir>/<technique>/results.pt  state computed by the code (model params, buffers, EMA
                                      teacher), engine activations, outputs, targets, losses,
                                      gradients
    <ref_dir>/<technique>/meta.json   commit, library versions, sizes

The `compare` command (and numerics_test.py) re-runs the step with the current code from the
stored config, batch and weights, and compares every tensor with bf16 tolerances.
The reference files are large and must not be committed.

Not covered, by design or for size (extend the configs or the harness to cover them):
- The data pipeline (masking, tokenisation, normalisation, teacher_time_offset): the batch is
  drawn once and stored, so the test isolates the model and loss code from data changes.
- Reductions over samples, streams and ensemble members: one sample, one stream (ERA5) and
  ensemble size 1. Several config values are identical (all level weights, the number of heads
  and the block factors of the engines), so wiring bugs between them can pass.
- Trainer.train: NumericsStep.run copies its forward/backward part, see there. Moving that part
  into a Trainer method that both call would cover changes to trainer.py as well.
- EMA updates, optimizer steps and any step other than the first (istep 0).

This must run on a GPU node (flash attention is required):

    uv run python integration_tests/numerics/numerics.py generate --ref-dir <dir>
    uv run python integration_tests/numerics/numerics.py compare --ref-dir <dir>
    WEATHERGEN_NUMERICS_REF_DIR=<dir> uv run pytest integration_tests/numerics/numerics_test.py
"""

import argparse
import dataclasses
import functools
import io
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

import weathergen.common.config as config
import weathergen.model.attention as attention
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.model_interface import init_model_and_shard
from weathergen.model.norms import RMSNorm
from weathergen.model.utils import apply_fct_to_blocks, set_to_eval
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.target_and_aux_diffusion import DiffusionLatentTargetEncoder
from weathergen.train.target_and_aux_ssl_teacher import EMATeacher, EncoderTeacher
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, extract_batch_metadata, get_target_idxs_from_cfg

# deterministic cuBLAS, required by torch.use_deterministic_algorithms (set before CUDA init)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

logger = logging.getLogger(__name__)

NUMERICS_DIR = Path(__file__).parent
CONFIGS_DIR = NUMERICS_DIR / "configs"
WEATHERGEN_HOME = NUMERICS_DIR.parent.parent

TECHNIQUES = [
    "forecasting",
    "diffusion",
    "diffusion_forecast_cond",
    "jepa_spatial",
    "jepa_temporal",
]

SEED = 42

# rtol is the bf16 default of torch.testing.assert_close; the absolute tolerance of each tensor
# scales with its magnitude, atol_t = max(ATOL, ATOL_SCALE * max|e_t|). The floor ATOL is far
# below the bf16 default of 1e-5, since most gradients of the tiny model are smaller than 1e-2
# and the element-wise check would otherwise be inactive for them.
RTOL = 1.6e-2
ATOL = 1e-8
ATOL_SCALE = float(os.environ.get("WEATHERGEN_NUMERICS_ATOL_SCALE", "1e-3"))
# norm check: ||a - e|| / ||e|| <= L2_TOL for each tensor with ||e|| > 0
L2_TOL = float(os.environ.get("WEATHERGEN_NUMERICS_L2_TOL", "1e-2"))

# all trainable parameters are re-drawn for the reference (see NumericsStep.reinit_params) so that
# no path is switched off by a zero initialisation; weights of NORMS are drawn around one.
# With 0.05 (vs. 0.02) the residual branches are larger, so mutations of them are detected in
# about 2-10x more tensors.
REINIT_STD = 0.05
NORMS = (torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.GroupNorm, RMSNorm)

# student modules whose outputs are recorded, to localise where a mismatch starts
HOOKED_MODULES = [
    r"encoder\.embed_engine",
    r"encoder\.ae_local_engine",
    r"encoder\.ae_local_global_engine",
    r"encoder\.ae_aggregation_engine",
    r"encoder\.ae_global_engine",
    r"forecast_engine",
    r"target_token_engines\.[^.]+",
    r"pred_heads\.[^.]+",
    r"latent_heads\.[^.]+",
    r"deep_ssl_fusion",
]

# order of the result groups in reports, in execution order to see where a deviation starts
GROUPS = ["state", "activations", "outputs", "losses", "grads"]

SIZE_LIMIT_BYTES = 1 << 30


def load_test_config(technique: str) -> config.Config:
    """Test config (base + technique overlay + streams), without the private config."""
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIGS_DIR / "numerics_base.yml"),
        OmegaConf.load(CONFIGS_DIR / f"{technique}.yml"),
    )
    cfg.streams = config.load_streams(NUMERICS_DIR / cfg.streams_directory)
    return cfg


def _seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


@contextmanager
def _isolated_shared_dir():
    """Redirect run/model directories created by the trainer to a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="wg_numerics_") as tmp:
        with mock.patch.object(config, "_get_shared_wg_path", lambda: Path(tmp)):
            yield


@contextmanager
def _deterministic_kernels(enabled: bool = True):
    """
    Use deterministic kernels so that repeated runs of the same code are bitwise identical.
    The flash-attn backward pass otherwise accumulates dq with atomics, which perturbs single
    gradient elements beyond bf16 tolerance between runs.
    """
    if not enabled:
        yield
        return
    deterministic_fcts = {
        name: functools.partial(getattr(attention, name), deterministic=True)
        for name in ("flash_attn_func", "flash_attn_varlen_func")
    }
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with mock.patch.multiple(attention, **deterministic_fcts):
            yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


@contextmanager
def _record_activations(model: torch.nn.Module):
    """
    Record the outputs of HOOKED_MODULES, per call, while the context is active. The hooks are
    removed on exit, so the recomputation of activation checkpointing during backward is not
    recorded.
    """
    activations: dict[str, torch.Tensor] = {}
    calls: dict[str, int] = {}

    def hook(name, module, args, output):
        i = calls.get(name, 0)
        calls[name] = i + 1
        activations.update(_flatten(output, f"{name}/call{i}"))

    pattern = re.compile("|".join(f"(?:{p})" for p in HOOKED_MODULES))
    handles = [
        module.register_forward_hook(functools.partial(hook, name))
        for name, module in model.named_modules()
        if pattern.fullmatch(name)
    ]
    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _disable_dropout(module: torch.nn.Module) -> None:
    """
    Set all dropout rates to 0. Some are hard-coded and ignore the config (e.g. 0.1 in
    model/embeddings.py and the decoder blocks); dropout masks would make the results depend on
    the order of all draws from the global RNG.
    """
    for m in module.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.p = 0.0
        # attention modules pass their rate to flash-attn instead of using a Dropout module
        if isinstance(getattr(m, "dropout_rate", None), float):
            m.dropout_rate = 0.0


class NumericsStep:
    """Builds the training components for one technique and runs a single training step."""

    def __init__(self, test_cfg: config.Config, technique: str):
        devices = Trainer.init_torch()
        # the test config replaces config/default_config.yml as the base
        cf = config.load_merge_configs(base=OmegaConf.create(test_cfg))
        cf = config.set_run_id(cf, f"numerics_{technique}", False)
        cf.stage = "train"
        self.cf = Trainer.init_ddp(cf)
        assert self.cf.world_size == 1, "The numerics test runs on a single GPU."

        self.trainer = Trainer(self.cf.train_logging)
        self.trainer.init(self.cf, devices)
        self.trainer.device = torch.device(f"cuda:{self.cf.local_rank}")

        _seed_all()
        self.trainer.dataset = MultiStreamDataSampler(
            self.cf, self.trainer.training_cfg, stage=TRAIN
        )
        self.trainer.model, self.trainer.model_params = init_model_and_shard(
            self.cf,
            self.trainer.dataset,
            None,
            None,
            self.trainer.training_cfg.training_mode,
            devices[0],
            self.cf.with_ddp,
            self.cf.with_fsdp,
        )
        self.trainer.target_and_aux_calculators = self.trainer.get_target_aux_calculators(
            self.trainer.training_cfg
        )
        self.trainer.loss_calculator = LossCalculator(
            self.cf, self.trainer.training_cfg, TRAIN, device=self.trainer.device
        )
        for module in self.modules().values():
            _disable_dropout(module)

    def draw_batch(self) -> bytes:
        """Draw the first batch of the (unshuffled, seeded) training dataset, serialized."""
        buffer = io.BytesIO()
        torch.save(next(iter(self.trainer.dataset)), buffer)
        return buffer.getvalue()

    def reinit_params(self) -> list[str]:
        """
        Re-draw all trainable student parameters from a seeded normal(0, REINIT_STD), and the
        weights of normalisation layers from 1 + normal(0, REINIT_STD), since zero-initialised
        parameters (e.g. adaLN gates) would make the gradients behind them vanish. Frozen
        parameters (positional encodings) are kept and EMA teachers are reset to the student.
        Returns the names of the re-drawn normalisation weights.
        """
        norm_weights = {
            f"{name}.weight"
            for name, module in self.trainer.model.named_modules()
            if isinstance(module, NORMS)
        }
        generator = torch.Generator().manual_seed(SEED)
        redrawn = []
        with torch.no_grad():
            for name, p in self.trainer.model.named_parameters():
                if p.requires_grad:
                    sample = REINIT_STD * torch.randn(p.shape, generator=generator)
                    p.copy_(1.0 + sample if name in norm_weights else sample)
                    redrawn.append(name)
        self.reset_teachers()
        return sorted(norm_weights.intersection(redrawn))

    def reset_teachers(self) -> None:
        """Reset the EMA teachers to copies of the student, as at the start of training."""
        for calc in self.trainer.target_and_aux_calculators.values():
            if isinstance(calc, EMATeacher):
                calc.reset()

    def modules(self) -> dict[str, torch.nn.Module]:
        """All modules of the step: student, model params, teachers, target encoders."""
        modules = {"model": self.trainer.model, "model_params": self.trainer.model_params}
        for name, calc in self.trainer.target_and_aux_calculators.items():
            if isinstance(calc, EMATeacher):
                modules[f"target_aux.{name}.teacher"] = calc.ema_model.ema_model
            elif isinstance(calc, DiffusionLatentTargetEncoder):
                modules[f"target_aux.{name}.encoder"] = calc.encoder
            if isinstance(calc, EncoderTeacher):
                for k, module in calc.postprocess_targets.items():
                    modules[f"target_aux.{name}.postprocess.{k}"] = module
        return modules

    def state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        The weights that are inputs of the step: the trainable student parameters (re-drawn
        for the reference) and the frozen diffusion target encoders, which stand in for
        weights loaded from a checkpoint. Everything else the code computes itself (positional
        encodings, neighbour indices, buffers, EMA teachers) and is compared in `computed_state`.
        """
        state_dicts = {
            "model": {
                name: p.detach().cpu().clone()
                for name, p in self.trainer.model.named_parameters()
                if p.requires_grad
            }
        }
        for name, module in self.modules().items():
            if name.endswith(".encoder"):
                state_dicts[name] = {
                    k: v.detach().cpu().clone() for k, v in module.state_dict().items()
                }
        return state_dicts

    def load_state_dicts(self, state_dicts: dict[str, dict[str, torch.Tensor]]) -> None:
        modules = self.modules()
        loaded = {"model"} | {name for name in modules if name.endswith(".encoder")}
        if loaded != set(state_dicts):
            raise ValueError(
                f"Loaded modules differ from the reference: current={sorted(loaded)}, "
                f"reference={sorted(state_dicts)}"
            )
        # renamed, missing or new trainable parameters must fail the test
        params = {n: p for n, p in self.trainer.model.named_parameters() if p.requires_grad}
        if set(params) != set(state_dicts["model"]):
            raise ValueError(
                "Trainable parameters differ from the reference: "
                f"new={sorted(set(params) - set(state_dicts['model']))}, "
                f"missing={sorted(set(state_dicts['model']) - set(params))}"
            )
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(state_dicts["model"][name])
        for name in loaded - {"model"}:
            modules[name].load_state_dict(state_dicts[name], strict=True)
        self.reset_teachers()

    def computed_state(self) -> dict[str, torch.Tensor]:
        """Weights and buffers computed by the code rather than loaded from the reference."""
        state = {}
        for name, module in self.modules().items():
            if name.endswith(".encoder"):
                continue
            trainable = (
                {n for n, p in module.named_parameters() if p.requires_grad}
                if name == "model"
                else set()
            )
            for k, v in module.state_dict().items():
                if k not in trainable:
                    state[f"{name}.{k}"] = v
        return state

    def run(self, batch_bytes: bytes) -> dict[str, dict[str, torch.Tensor]]:
        """
        One training step: a copy of the forward, target, loss and backward part of
        Trainer.train, which has no separate step method. The loss-spike check, GradScaler,
        gradient clipping, optimizer step and logging are left out: they act on the gradients
        after they are computed, and the stored gradients are the unclipped ones.
        Keep this in sync with Trainer.train.
        """
        state = self.computed_state()
        batch = torch.load(io.BytesIO(batch_bytes), weights_only=False)
        batch.to_device(self.trainer.device)

        self.trainer.model.train()
        apply_fct_to_blocks(self.trainer.model, self.cf.freeze_modules, set_to_eval)
        self.trainer.model.zero_grad(set_to_none=True)

        _seed_all()
        with (
            _record_activations(self.trainer.model) as activations,
            torch.autocast(
                device_type=f"cuda:{self.cf.local_rank}",
                dtype=self.trainer.mixed_precision_dtype,
                enabled=self.cf.with_mixed_precision,
            ),
        ):
            preds = self.trainer.model(
                model_params=self.trainer.model_params,
                samples_or_output=batch.get_source_samples(),
                forecast_steps=batch.get_output_idxs(),
            )
            targets_and_auxs = {}
            for loss_name, target_aux in self.trainer.target_and_aux_calculators.items():
                target_idxs = get_target_idxs_from_cfg(self.trainer.training_cfg, loss_name)
                targets_and_auxs[loss_name] = target_aux.compute(
                    self.cf.general.istep,
                    batch.get_target_samples(target_idxs),
                    self.trainer.model_params,
                    self.trainer.model,
                )

        loss = self.trainer.loss_calculator.compute_loss(
            preds=preds,
            targets_and_aux=targets_and_auxs,
            metadata=extract_batch_metadata(batch),
            istep=self.cf.general.istep,
        )
        for target_aux in self.trainer.target_and_aux_calculators.values():
            target_aux.update_state_pre_backward(self.cf.general.istep, batch, self.trainer.model)
        loss.backward()

        losses = {"loss": loss.detach()}
        losses.update(_flatten(self.trainer.loss_calculator.losses_unweighted_hist[-1], "terms"))

        outputs = {}
        for field in ("physical", "latent", "latent_deep"):
            outputs.update(_flatten(getattr(preds, field, None), f"preds.{field}"))
        for loss_name, t_aux in targets_and_auxs.items():
            for field in ("physical", "latent", "latent_deep", "aux_outputs"):
                outputs.update(
                    _flatten(getattr(t_aux, field, None), f"targets.{loss_name}.{field}")
                )

        # parameters without a gradient are stored as empty tensors
        grads = {
            name: p.grad if p.grad is not None else torch.tensor([])
            for name, p in self.trainer.model.named_parameters()
            if p.requires_grad
        }

        results = {
            "state": state,
            "activations": activations,
            "outputs": outputs,
            "losses": losses,
            "grads": grads,
        }
        return {
            group: {k: v.detach().cpu().clone() for k, v in tensors.items()}
            for group, tensors in results.items()
        }


def _flatten(obj, prefix: str, depth: int = 0) -> dict[str, torch.Tensor]:
    """
    Flatten nested containers / dataclasses / plain objects into {path: tensor}.

    The path is the tensor's name in the reference: `prefix` accumulates the access path while
    descending, with `.key` for dict keys and attributes and `[i]` for list/tuple entries, e.g.
    `preds.latent[0].latent_state.patch_tokens`. Tensors of two runs are matched by this path.
    """
    out = {}
    # configs (e.g. the masking config in the target metadata) hold settings, not numerics
    if depth > 8 or obj is None or OmegaConf.is_config(obj):
        return out
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, np.ndarray):  # noqa: TID251
        # keep bool (b), signed (i) and unsigned (u) integer and float (f) arrays; skip arrays
        # torch can't hold or that are no numerics (object, string, datetime, complex)
        if obj.dtype.kind in "biuf":
            out[prefix] = torch.from_numpy(np.ascontiguousarray(obj))
    elif isinstance(obj, bool | int | float):
        out[prefix] = torch.tensor(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}", depth + 1))
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]", depth + 1))
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            out.update(_flatten(getattr(obj, f.name), f"{prefix}.{f.name}", depth + 1))
    elif hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        for k, v in vars(obj).items():
            out.update(_flatten(v, f"{prefix}.{k}", depth + 1))
    return out


def _nbytes(tensors) -> int:
    if isinstance(tensors, torch.Tensor):
        return tensors.numel() * tensors.element_size()
    if isinstance(tensors, bytes):
        return len(tensors)
    if isinstance(tensors, dict):
        return sum(_nbytes(v) for v in tensors.values())
    return 0


def _environment() -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(WEATHERGEN_HOME), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    try:
        import flash_attn

        flash_version = flash_attn.__version__
    except ImportError:
        flash_version = None
    return {
        "commit": commit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_version,
        "gpu": torch.cuda.get_device_name(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }


def generate(
    technique: str, ref_dir: Path, overwrite: bool = False, deterministic: bool = True
) -> dict:
    """Run the step on the current code and store it as the golden reference."""
    out_dir = ref_dir / technique
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace the reference.")

    test_cfg = load_test_config(technique)
    with _isolated_shared_dir(), _deterministic_kernels(deterministic):
        step = NumericsStep(test_cfg, technique)
        batch_bytes = step.draw_batch()
        norm_weights = step.reinit_params()
        state_dicts = step.state_dicts()
        results = step.run(batch_bytes)
        num_params = {
            name: {
                "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "total": sum(p.numel() for p in module.parameters()),
            }
            for name, module in step.modules().items()
        }

    inputs = {
        "technique": technique,
        "seed": SEED,
        "deterministic": deterministic,
        "config": OmegaConf.to_container(test_cfg, resolve=True),
        "batch": batch_bytes,
        "state_dicts": state_dicts,
    }
    sizes = {"batch": len(batch_bytes), "weights": _nbytes(state_dicts)}
    sizes.update({group: _nbytes(tensors) for group, tensors in results.items()})
    for group, size in sizes.items():
        if size > SIZE_LIMIT_BYTES:
            logger.warning(f"{technique}: {group} is {size / 2**20:.1f} MiB (> 1 GiB).")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, out_dir / "inputs.pt")
    torch.save(results, out_dir / "results.pt")
    meta = {
        "technique": technique,
        "deterministic": deterministic,
        "environment": _environment(),
        "sizes_bytes": sizes,
        "num_params": num_params,
        "num_tensors": {group: len(tensors) for group, tensors in results.items()},
        "reinit": {"std": REINIT_STD, "norm_weights_around_one": norm_weights},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def rerun(technique: str, ref_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Run the step on the current code from the reference config, batch and weights."""
    inputs = torch.load(ref_dir / technique / "inputs.pt", weights_only=False)
    with _isolated_shared_dir(), _deterministic_kernels(inputs["deterministic"]):
        step = NumericsStep(OmegaConf.create(inputs["config"]), technique)
        step.load_state_dicts(inputs["state_dicts"])
        return step.run(inputs["batch"])


@dataclasses.dataclass
class Mismatch:
    group: str
    name: str
    reason: str
    max_abs_err: float = float("nan")
    max_rel_err: float = float("nan")
    rel_l2_err: float = float("nan")
    atol: float = float("nan")
    num_bad: int = 0
    numel: int = 0


def _max(t: torch.Tensor) -> float:
    return t.max().item() if t.numel() > 0 else 0.0


def compare_results(
    actual: dict, expected: dict, atol_scale: float = ATOL_SCALE, l2_tol: float = L2_TOL
) -> tuple[list[Mismatch], dict]:
    """
    A float tensor passes if (1) element-wise |a - e| <= atol_t + RTOL * |e|, with the
    per-tensor atol_t = max(ATOL, atol_scale * max|e|) over the finite entries, (2) non-finite
    entries (NaN, +inf, -inf) match exactly, and (3) the norm check ||a - e|| / ||e|| <= l2_tol
    holds over the finite entries (for ||e|| > 0; all-zero references rely on (1)). Integer and
    bool tensors must be equal.
    """
    mismatches = []
    stats = {}
    for group, exp_tensors in expected.items():
        act_tensors = actual.get(group, {})
        for name in sorted(set(exp_tensors) | set(act_tensors)):
            if name not in act_tensors:
                mismatches.append(Mismatch(group, name, "missing in current code"))
                continue
            if name not in exp_tensors:
                mismatches.append(Mismatch(group, name, "not in reference"))
                continue
            a, e = act_tensors[name], exp_tensors[name]
            if a.shape != e.shape:
                reason = f"shape {tuple(a.shape)} != {tuple(e.shape)}"
                mismatches.append(Mismatch(group, name, reason))
                continue
            if a.dtype != e.dtype:
                mismatches.append(Mismatch(group, name, f"dtype {a.dtype} != {e.dtype}"))
                continue
            if a.numel() == 0:
                continue
            if not (a.is_floating_point() or a.is_complex()):
                if not torch.equal(a, e):
                    num_bad = int((a != e).sum().item())
                    mismatches.append(
                        Mismatch(
                            group, name, "values differ (exact)", num_bad=num_bad, numel=a.numel()
                        )
                    )
                continue
            a, e = a.double(), e.double()
            fin_a, fin_e = torch.isfinite(a), torch.isfinite(e)
            finite = fin_a & fin_e
            diff = torch.where(finite, (a - e).abs(), 0.0)
            atol = max(ATOL, atol_scale * _max(e[finite].abs()))
            bad = diff > atol + RTOL * torch.where(finite, e.abs(), 0.0)
            # NaN, +inf and -inf must occur at the same positions
            same_nonfinite = (a == e) | (torch.isnan(a) & torch.isnan(e))
            bad |= ~finite & ~same_nonfinite
            max_abs = _max(diff[finite])
            max_rel = _max(diff[finite] / e[finite].abs().clamp_min(atol))
            e_norm = torch.linalg.vector_norm(e[finite]).item()
            diff_norm = torch.linalg.vector_norm(diff[finite]).item()
            rel_l2 = diff_norm / e_norm if e_norm > 0 else (0.0 if diff_norm == 0 else math.inf)
            stats[f"{group}/{name}"] = {
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "rel_l2_err": rel_l2,
                "atol": atol,
            }
            fails = []
            if bad.any():
                fails.append("element-wise")
            if e_norm > 0 and rel_l2 > l2_tol:
                fails.append("rel L2")
            if fails:
                mismatches.append(
                    Mismatch(
                        group,
                        name,
                        f"values differ ({' and '.join(fails)})",
                        max_abs,
                        max_rel,
                        rel_l2,
                        atol,
                        int(bad.sum().item()),
                        a.numel(),
                    )
                )
    return mismatches, stats


def format_mismatches(technique: str, mismatches: list[Mismatch], max_lines: int = 40) -> str:
    lines = [
        f"{technique}: {len(mismatches)} tensors outside bf16 tolerance (rtol={RTOL}, "
        f"atol=max({ATOL}, {ATOL_SCALE} * max|ref|), rel L2 <= {L2_TOL}). "
        "Activations are listed in execution order:"
    ]
    for m in mismatches[:max_lines]:
        line = f"  [{m.group}] {m.name}: {m.reason}"
        if m.numel and not math.isnan(m.max_abs_err):
            line += (
                f" ({m.num_bad}/{m.numel} elements, max abs err {m.max_abs_err:.3e} "
                f"(atol {m.atol:.1e}), max rel err {m.max_rel_err:.3e}, "
                f"rel L2 err {m.rel_l2_err:.3e})"
            )
        lines.append(line)
    if len(mismatches) > max_lines:
        lines.append(f"  ... and {len(mismatches) - max_lines} more")
    return "\n".join(lines)


def compare(technique: str, ref_dir: Path, report_path: Path | None = None) -> list[Mismatch]:
    expected = torch.load(ref_dir / technique / "results.pt", weights_only=False)
    actual = rerun(technique, ref_dir)
    mismatches, stats = compare_results(actual, expected)
    # sort by group, and activations in recorded (execution) order (the sort is stable)
    order = {k: i for i, k in enumerate(expected.get("activations", {}))}
    mismatches.sort(key=lambda m: (GROUPS.index(m.group), order.get(m.name, 0)))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "technique": technique,
            "environment": _environment(),
            "mismatches": [dataclasses.asdict(m) for m in mismatches],
            "stats": stats,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    return mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("command", choices=["generate", "compare"])
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--technique", choices=TECHNIQUES + ["all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="replace existing references")
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="generate: use the default (non-deterministic) flash-attn backward",
    )
    parser.add_argument("--report-dir", type=Path, default=None, help="write JSON reports here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    techniques = TECHNIQUES if args.technique == "all" else [args.technique]
    failed = False
    for technique in techniques:
        if args.command == "generate":
            meta = generate(technique, args.ref_dir, args.overwrite, not args.nondeterministic)
            sizes = ", ".join(f"{k}={v / 2**20:.1f} MiB" for k, v in meta["sizes_bytes"].items())
            print(f"{technique}: reference written ({sizes})")
        else:
            report = args.report_dir / f"{technique}.json" if args.report_dir else None
            mismatches = compare(technique, args.ref_dir, report)
            failed |= bool(mismatches)
            print(format_mismatches(technique, mismatches) if mismatches else f"{technique}: PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
