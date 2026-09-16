#!/usr/bin/env python3
"""Generate a fresh diffusion-training config: forks a backbone's encoder (+ forecast engine, if
it already has one) via `load_chkpt`, freezes everything except the forecast engine ("FE training
only"), and trains a new diffusion forecast engine on top. This is the *first* stage in the
lineage -- distinct from `weathergen-dcft`, which finetunes a decoder for a run this stage already
produced, and from `weathergen-decoder-overlay`, which borrows a decoder at inference time only.

Unlike a decoder finetune (where there's one standard recipe), a fresh diffusion-training config
is a genuine experiment-design choice -- which backbone, what noise schedule, what worker counts,
what EMA -- so this script does not invent hyperparameters. It takes a `--template`: a prior
diffusion-training run this new experiment is modeled after ("same as X, but change one knob" is
how every such experiment in this repo's history is actually described), copies its full resolved
config (training/validation knobs, diffusion hyperparameters -- zero drift by construction), then
applies whatever this experiment changes via `--set key=value` (same dotlist syntax launch-slurm.py
itself uses for --options). Architecture is always copied from the *backbone* being forked from
(required for load_chkpt weight-shape compatibility), not from the template, in case they differ.

See ../references/diffusion-training-notes.md for the reasoning and known gotchas (the missing
--chain-jobs stall, the stale streams_directory HPC-mount gotcha).

Run from the repo root, under the project environment (checkpoint inspection needs torch):

    uv run python .claude/skills/weathergen-diffusion-training/scripts/make_diffusion_training_config.py \
        --template <PRIOR-RUN-ID> --tag <short-experiment-name> \
        [--backbone <BACKBONE-RUN-ID>] [--backbone-mini-epoch <N>] \
        [--set key=value ...]

This only writes a config file and prints a suggested launch command -- it never submits anything.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# "FE training only": freezes encoder + decoder (if any), leaves the forecast engine trainable.
# Verified against real checkpoint keys below, not trusted blindly -- see FREEZE_TRAINABLE_PREFIX.
FREEZE_FE_ONLY = (
    r".*latent_pre_norm.*|.*latent_heads.*|.*pred_heads.*|.*target_token_engines.*|"
    r".*embed_target_coords.*|.*encoder.*|.*StreamEmbedder_ERA5.*|.*embed_engine.*|"
    r".*ae_local_engine.*|.*ae_local_global_engine.*|.*ae_global_engine.*"
)
FREEZE_TRAINABLE_PREFIX = "forecast_engine"

# Fields that must come from the run being forked from (load_chkpt's backbone), not the template:
# everything that determines the shape of the weights `load_chkpt` actually loads -- i.e. the
# encoder + decoder, since "FE training only" freezes and loads exactly those. Only consulted when
# --backbone differs from the template's own load_chkpt.run_id -- the common case (a new experiment
# variant on the *same* backbone as its template) never needs this at all.
#
# `fe_*` is deliberately NOT here. The forecast engine is trained *fresh* -- nothing loads into
# `forecast_engine.*` (a JEPA/masking backbone's plain FE keys mismatch the diffusion FE and land
# as "not found in model" warnings), so its depth/width/dropout must track the *template's*
# diffusion FE, not the backbone's. Copying `fe_num_blocks` from the backbone here once silently
# gave a 16-block diffusion FE where the template (and every sibling run) used 12.
ARCHITECTURE_KEYS = [
    "embed_orientation",
    "embed_unembed_mode",
    "embed_dropout_rate",
    "ae_local_dim_embed",
    "ae_local_num_blocks",
    "ae_local_num_heads",
    "ae_local_dropout_rate",
    "ae_local_with_qk_lnorm",
    "ae_local_num_queries",
    "ae_local_queries_per_cell",
    "ae_adapter_num_heads",
    "ae_adapter_embed",
    "ae_adapter_with_qk_lnorm",
    "ae_adapter_with_residual",
    "ae_adapter_dropout_rate",
    "ae_global_dim_embed",
    "ae_global_num_blocks",
    "ae_global_num_heads",
    "ae_global_dropout_rate",
    "ae_global_with_qk_lnorm",
    "ae_global_att_dense_rate",
    "ae_global_block_factor",
    "ae_global_mlp_hidden_factor",
    "ae_global_trailing_layer_norm",
    "ae_aggregation_num_blocks",
    "ae_aggregation_num_heads",
    "ae_aggregation_dropout_rate",
    "ae_aggregation_with_qk_lnorm",
    "ae_aggregation_att_dense_rate",
    "ae_aggregation_block_factor",
    "ae_aggregation_mlp_hidden_factor",
    "decoder_type",
    "pred_adapter_kv",
    "pred_self_attention",
    "pred_dyadic_dims",
    "pred_mlp_adaln",
    "num_class_tokens",
    "num_register_tokens",
    "healpix_level",
    "rope_2D",
    "mlp_type",
    "use_xsa",
    "norm_type",
    "qk_norm_type",
]

RESET_GENERAL = {
    "istep": 0,
    "rank": "???",
    "world_size": "???",
    "multiprocessing_method": "fork",
    "desc": "",
    "run_id": "???",
    "run_history": [],
}

# Same rationale as weathergen-dcft's script: these are the *template* run's own stale
# launch-environment/MLflow artifacts, not config a fresh launch should inherit.
DENY_TOP_LEVEL = {
    "mlflow",
    "stage",
    "from_run_id",
    "local_rank",
    "with_ddp",
    "world_size_original",
    "world_size",
    "rank",
    # Host-specific storage roots. A template that ran on another HPC carries that machine's
    # absolute paths (e.g. ni41n7gy ships /iopsstor/scratch/cscs/... from CSCS), and the private
    # config does NOT override them -- paths.yml defines path_shared_working_dir /
    # path_shared_slurm_dir instead. model_path is live: teacher_utils.py does
    # cf.get("model_path", get_path_model(...)), so a stale value silently replaces the correct
    # location on the SSL-teacher path. Drop both; absent means the code derives them correctly
    # from the private config. See the 2026-09-10 new73gkw incident.
    "model_path",
    "run_path",
}
DENY_TOP_LEVEL_PREFIXES = ("data_path", "path_shared_")


def die(msg: str) -> None:
    sys.exit(f"error: {msg}")


def run_config(models_dir: Path, run_id: str) -> dict:
    for name in (f"model_{run_id}.json", f"model_{run_id}_latest.json"):
        path = models_dir / run_id / name
        if path.is_file():
            return json.loads(path.read_text())
    die(f"no model config for {run_id!r} under {models_dir / run_id}")


def checkpoint_path(models_dir: Path, run_id: str, mini_epoch: int) -> Path:
    tag = "latest" if mini_epoch in (-1, None) else f"chkpt{mini_epoch:05d}"
    path = models_dir / run_id / f"{run_id}_{tag}.chkpt"
    if not path.is_file():
        die(f"no checkpoint {path}")
    return path


def state_dict_keys(path: Path) -> list[str]:
    import torch  # local: only this step needs the project environment

    state = torch.load(path, map_location="meta", mmap=True, weights_only=True)
    return [k[len("module.") :] if k.startswith("module.") else k for k in state.keys()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--template",
        required=True,
        help="prior diffusion-training run to copy hyperparameters from",
    )
    ap.add_argument(
        "--tag", required=True, help="short name for this experiment, used in the output filename"
    )
    ap.add_argument(
        "--backbone", help="run id to fork from (default: template's own load_chkpt.run_id)"
    )
    ap.add_argument(
        "--backbone-mini-epoch",
        type=int,
        default=None,
        help="backbone checkpoint to fork from (default: template's own load_chkpt.mini_epoch if "
        "--backbone matches the template's backbone, else required)",
    )
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="dotlist override applied on top of the template, e.g. --set sigma_max=0.9 --set "
        "data_loading.num_workers=4 (repeatable; same syntax as launch-slurm.py's --options)",
    )
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="default: config/diffusion/config_diffusion_<BACKBONE>_<TAG>.yml",
    )
    args = ap.parse_args()

    template_cfg = run_config(args.models_dir, args.template)
    template_backbone = (template_cfg.get("load_chkpt") or {}).get("run_id")
    if not template_backbone:
        die(f"template {args.template!r} has no load_chkpt; can't infer its own backbone")

    backbone = args.backbone or template_backbone
    backbone_mini_epoch = args.backbone_mini_epoch
    if backbone_mini_epoch is None:
        if backbone == template_backbone:
            backbone_mini_epoch = template_cfg["load_chkpt"]["mini_epoch"]
        else:
            die(
                "--backbone differs from the template's own backbone; --backbone-mini-epoch is required"
            )
    run_config(args.models_dir, backbone)  # existence check

    backbone_keys = state_dict_keys(checkpoint_path(args.models_dir, backbone, backbone_mini_epoch))
    pattern = re.compile(FREEZE_FE_ONLY)
    should_freeze = [k for k in backbone_keys if not k.startswith(FREEZE_TRAINABLE_PREFIX)]
    should_train = [k for k in backbone_keys if k.startswith(FREEZE_TRAINABLE_PREFIX)]
    not_frozen = [k for k in should_freeze if not pattern.search(k)]
    if not_frozen:
        die(
            f"the standard 'FE training only' freeze pattern does not match {len(not_frozen)} "
            f"non-forecast_engine key(s) from {backbone!r}, e.g. {not_frozen[0]!r} -- this "
            "architecture has a module family the standard pattern doesn't know about. Inspect "
            "the key and write a config by hand with a corrected freeze_modules instead."
        )
    wrongly_frozen = [k for k in should_train if pattern.search(k)]
    if wrongly_frozen:
        _logger_msg = (
            f"warning: the freeze pattern also matches {len(wrongly_frozen)} forecast_engine "
            f"key(s), e.g. {wrongly_frozen[0]!r} -- these would stay frozen too, which may not "
            "be intended for 'FE training'. Continuing, but check freeze_modules in the output."
        )
        print(_logger_msg, file=sys.stderr)

    cfg = {
        k: copy.deepcopy(v)
        for k, v in template_cfg.items()
        if k not in DENY_TOP_LEVEL and not k.startswith(DENY_TOP_LEVEL_PREFIXES)
    }

    if backbone != template_backbone:
        backbone_cfg = run_config(args.models_dir, backbone)
        for key in ARCHITECTURE_KEYS:
            if key in backbone_cfg:
                cfg[key] = copy.deepcopy(backbone_cfg[key])
        print(
            f"backbone {backbone!r} differs from template {args.template!r}'s own backbone "
            f"{template_backbone!r} -- architecture fields re-copied from {backbone!r} "
            "(everything else still from the template)."
        )

    cfg["freeze_modules"] = FREEZE_FE_ONLY
    cfg["load_chkpt"] = {"run_id": backbone, "mini_epoch": backbone_mini_epoch}
    cfg.pop("load_decoder_chkpt", None)  # a fresh FE-training run never borrows a decoder
    cfg["streams"] = "???"
    cfg["general"] = dict(RESET_GENERAL)
    cfg.setdefault("data_loading", {})["rng_seed"] = "???"
    cfg["wgtags"] = {"org": None, "issue": None, "exp": None, "grid": None}

    if args.overrides:
        overlay = OmegaConf.to_container(OmegaConf.from_dotlist(args.overrides), resolve=True)
        cfg = OmegaConf.to_container(OmegaConf.merge(cfg, overlay), resolve=True)

    # The noise spec exists in up to three places in the resolved config (no live YAML anchor once
    # loaded from JSON): top-level, and duplicated into model_input's *and* target_input's
    # masking_strategy_config. Force both nested copies to track the top level regardless of how it
    # got set (template default or --set), so "change sigma" can never mean "changed it in the one
    # place that isn't actually read for the noise sampling that matters.
    #
    # target_input in particular ships from some templates as only {diffusion_rn: True} -- i.e. with
    # no noise_distribution/sigma at all -- so it must be *populated*, not merely synced, or the
    # denoising target is drawn from a different noise spec than the model input. See the
    # 2026-09-10 target_input fix.
    _noise_keys = ("noise_distribution", "sigma_min", "sigma_max")
    for _side in ("model_input", "target_input"):
        _msc = (
            cfg.get("training_config", {})
            .get(_side, {})
            .get("forecasting", {})
            .get("masking_strategy_config")
        )
        if _msc is None:
            continue
        for _k in _noise_keys:
            if _k in cfg:
                _msc[_k] = cfg[_k]

    out = (
        args.output or Path("config") / "diffusion" / f"config_diffusion_{backbone}_{args.tag}.yml"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    dropped = sorted(
        k for k in template_cfg if k not in cfg and k != "streams" and k != "load_decoder_chkpt"
    )
    header = (
        f"# Fresh diffusion-training config, generated by make_diffusion_training_config.py.\n"
        f"# Modeled on {args.template} (hyperparameters), forked from {backbone}@{backbone_mini_epoch} "
        f"(architecture + load_chkpt) -- see the skill's references/diffusion-training-notes.md.\n"
        f"# freeze_modules verified against the real checkpoint keys of {backbone!r}: "
        f"{len(should_freeze) - len(not_frozen)}/{len(should_freeze)} non-forecast_engine keys "
        f"frozen, {len(should_train) - len(wrongly_frozen)}/{len(should_train)} forecast_engine "
        f"keys left trainable.\n"
        f"# Dropped as stale launch-environment/MLflow fields from the template: "
        f"{', '.join(dropped) or '(none)'}.\n"
        f"# Overrides applied via --set: {', '.join(args.overrides) or '(none)'}.\n"
        f"# Before launching: re-verify streams_directory resolves on this filesystem (a common "
        f"gotcha -- see references/diffusion-training-notes.md), and start_date/end_date/"
        f"num_mini_epochs/samples_per_mini_epoch are what this experiment actually wants.\n"
    )
    with out.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, width=100)

    print(f"wrote {out}")
    print(
        f"  freeze_modules verified: {len(should_freeze) - len(not_frozen)}/{len(should_freeze)} frozen, "
        f"{len(should_train) - len(wrongly_frozen)}/{len(should_train)} forecast_engine keys trainable"
    )
    print(f"  dropped stale fields: {', '.join(dropped) or '(none)'}")
    print("\nsuggested launch (do not run automatically -- see SKILL.md):")
    print(
        f"  ../WeatherGenerator-private/hpc/launch-slurm.py --nodes 2 --chain-jobs 4 --base-config {out}"
    )
    print(
        "  (--chain-jobs 4 is not optional here -- a run this launches with --chain-jobs omitted "
        "(defaults to 1) will silently stop after ~12h with no further job queued; see "
        "references/diffusion-training-notes.md)"
    )


if __name__ == "__main__":
    main()
