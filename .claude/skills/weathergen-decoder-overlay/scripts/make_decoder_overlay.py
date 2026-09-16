#!/usr/bin/env python3
"""Generate an inference overlay config that attaches a backbone's physical decoder to a
run that has none (typically a run trained with a latent-only loss).

Writes the four blocks documented in ../references/decoder-overlay.md:

  1. a zero-weight LossPhysical in validation_config  -> makes the decoder modules exist
  2. load_decoder_chkpt                               -> supplies the decoder weights
  3. the backbone's decoded stream, verbatim          -> makes the shapes and the name match
     plus reconstruct: false on the model's own target-carrying streams
  4. the offset=1 / num_steps_input=1 block in test_config (always, inference only) -> aligns
     a diffusion model's single forecast step with a deterministic offset=1 model's lead time

Run it from the repo root (so that `models/` resolves), under the project environment because
the checkpoint inspection needs torch:

    uv run python .claude/skills/weathergen-inference-diagnostics/scripts/make_decoder_overlay.py \
        --model <MODEL-RUN-ID> [--backbone <RUN-ID>] [--decoded-stream <NAME>]

With no --backbone it uses the model's own `load_chkpt.run_id`. With no --decoded-stream it takes
the stream the backbone's decoder weights are keyed to, and refuses if that is ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

DECODER_PREFIXES = ("embed_target_coords", "target_token_engines", "pred_heads")
# get_targets_coords_size(): geoinfo + 5*(3*5) + 3*8 + 6
COORD_SIZE_OFFSET = 5 * (3 * 5) + 3 * 8 + 6


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


def decoder_shapes(path: Path) -> dict[str, dict[str, tuple]]:
    """Map stream name -> {parameter name: shape} for every decoder parameter."""
    import torch  # local: only this step needs the project environment

    state = torch.load(path, map_location="meta", mmap=True, weights_only=True)
    found: dict[str, dict[str, tuple]] = {}
    for key, value in state.items():
        bare = key[len("module.") :] if key.startswith("module.") else key
        if bare.startswith(DECODER_PREFIXES):
            stream = bare.split(".")[1]
            found.setdefault(stream, {})[bare] = tuple(value.shape)
    return found


def check_shapes(stream_cfg: dict, shapes: dict[str, tuple], stream: str) -> list[str]:
    """Compare the checkpoint's decoder shapes against the stream config it will be built from."""
    notes = []
    n_targets = len(stream_cfg.get("val_target_channels") or stream_cfg.get("target") or [])
    n_geoinfo = len(stream_cfg.get("geoinfo_channels") or [])

    for name, shape in shapes.items():
        if name.startswith("pred_heads") and name.endswith("0.0.weight"):
            if shape[0] != n_targets:
                notes.append(
                    f"pred head emits {shape[0]} channels but stream {stream!r} declares "
                    f"{n_targets} target channels"
                )
        if name.startswith("embed_target_coords") and name.endswith("weight"):
            expected = n_geoinfo + COORD_SIZE_OFFSET
            if shape[-1] != expected:
                notes.append(
                    f"coord embedding expects {shape[-1]} inputs but stream {stream!r} implies "
                    f"{expected} ({n_geoinfo} geoinfo channels + {COORD_SIZE_OFFSET})"
                )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run id of the model to run inference with")
    ap.add_argument("--backbone", help="run id providing the decoder (default: model's load_chkpt)")
    ap.add_argument("--decoded-stream", help="stream the decoder is keyed to (default: detected)")
    ap.add_argument("--mini-epoch", type=int, default=-1, help="backbone checkpoint to read")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("-o", "--output", type=Path, help="default: config/inference_decoder_overlay_<backbone>.yml")
    args = ap.parse_args()

    model_cfg = run_config(args.models_dir, args.model)

    backbone = args.backbone or (model_cfg.get("load_chkpt") or {}).get("run_id")
    if not backbone:
        die(f"{args.model} has no load_chkpt; pass --backbone explicitly")
    backbone_cfg = run_config(args.models_dir, backbone)

    shapes = decoder_shapes(checkpoint_path(args.models_dir, backbone, args.mini_epoch))
    if not shapes:
        die(f"backbone {backbone!r} has no decoder weights ({'/'.join(DECODER_PREFIXES)})")

    stream = args.decoded_stream
    if stream is None:
        if len(shapes) > 1:
            die(f"backbone {backbone!r} has decoders for {sorted(shapes)}; pass --decoded-stream")
        stream = next(iter(shapes))
    elif stream not in shapes:
        die(f"backbone {backbone!r} has no decoder for stream {stream!r} (has {sorted(shapes)})")

    if stream not in backbone_cfg.get("streams", {}):
        die(f"stream {stream!r} is in the checkpoint but not in the backbone's model config")
    stream_cfg = backbone_cfg["streams"][stream]

    for note in check_shapes(stream_cfg, shapes[stream], stream):
        print(f"warning: {note}", file=sys.stderr)

    # the model's own streams keep their targets but must not get a second, untrained decoder
    streams: dict[str, dict] = {}
    for name, cfg in model_cfg.get("streams", {}).items():
        if name == stream:
            continue
        if cfg.get("train_target_channels") or cfg.get("val_target_channels") or cfg.get("target"):
            streams[name] = {"reconstruct": False}
    streams[stream] = stream_cfg  # verbatim, including the derived channel lists
    # Force the decoded stream to be target-only on the source side. Some decoder-finetune
    # configs carry `source: null` and/or `{train,val}_source_channels: [~100 zarr channels]`
    # for this stream; the data reader uses `<stage>_source_channels` if present (else resolves
    # `source`), so leaving them feeds the encoder a spurious ~100-channel ERA5 source input it
    # was never trained on -> the diffusion rollout diverges. Conditioning comes through
    # <stream>_in, not <stream>. Zero out every source-channel field:
    for k in ("source", "train_source_channels", "val_source_channels"):
        if k in stream_cfg:
            stream_cfg[k] = []

    overlay = {
        "streams": streams,
        # Offset=1 / num_steps_input=1: relabels the model's sole loaded input step as "now"
        # and its forecast step(s) as starting +1 x time_step ahead, so the diffusion model's
        # RMSE rollout aligns with a deterministic (offset=1) model's lead times. Lives under
        # test_config -- inference only -- never training_config/validation_config, which
        # would wrongly apply the same relabeling during training/validation.
        # NOTE: every test_config field belongs here, not in --options. MLflow logs each
        # top-level config key as an immutable param, so setting test_config in the yml AND
        # again via --options raises INVALID_PARAMETER_VALUE at submission (unless
        # --no-register). Edit the values below rather than overriding them on the CLI.
        "test_config": {
            "model_input": {"forecasting": {"num_steps_input": 1}},
            "forecast": {"offset": 1, "policy": "fixed"},
            # MUST be >= world_size * batch_size * max(1, num_workers), and ideally a multiple:
            # multi_stream_data_sampler.py:238 integer-divides it by world_size, so 1 sample on a
            # 4-rank launch-slurm job (--nodes 1) gives len=0 -> no batches, metrics nan, no output
            # written, and check_samples still reports "sufficient available samples". 8 is safe
            # for 1/2/4/8 ranks; raise it for a real evaluation.
            "samples_per_mini_epoch": 8,
            "output": {"num_samples": 1},
        },
        # Same immutable-param rule as test_config above: keep every validation_config field
        # here, never in --options. validation_noise_levels=[] disables the extra fixed-noise
        # validation passes, which an inference rollout does not want.
        "validation_config": {
            "validation_noise_levels": [],
            "losses": {
                "physical": {
                    "type": "LossPhysical",
                    "weight": 0.0,
                    "target_and_aux_calc": "Physical",
                    "loss_fcts": {"mse": {}},
                }
            }
        },
        "load_decoder_chkpt": {"run_id": backbone, "mini_epoch": args.mini_epoch},
    }

    out = args.output or Path("config") / f"inference_decoder_overlay_{backbone}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Decoder overlay: run {args.model} with the physical decoder of {backbone}.\n"
        f"# Generated by make_decoder_overlay.py -- see the skill's references/decoder-overlay.md.\n"
        f"# Decoded stream: {stream}. The block below is copied verbatim from the backbone's\n"
        f"# model config, derived channel lists included: nothing recomputes them for an overlay.\n"
        f"# Includes the offset=1/num_steps_input=1 test_config block (always, inference only) --\n"
        f"# see Block 4 in references/decoder-overlay.md.\n"
        f"#\n"
        f"# Example launch (edit num_steps here / samples in test_config below before running;\n"
        f"# --wgen-dir runs the job from the current checkout, not the stored snapshot).\n"
        f"# MLflow logs each top-level key as one immutable param, so streams / test_config /\n"
        f"# validation_config must NOT also appear in --options: that fails submission with\n"
        f"# INVALID_PARAMETER_VALUE (--no-register skips the tracker and sidesteps it):\n"
        f"#\n"
        f"#   ../WeatherGenerator-private/hpc/launch-slurm.py \\\n"
        f"#     --stage inference \\\n"
        f"#     --from-run-id {args.model} \\\n"
        f"#     --wgen-dir $PWD \\\n"
        f"#     --nodes 1 \\\n"
        f"#     --config ./{out} \\\n"
        f"#     --options \\\n"
        f"#       training_config.forecast.num_steps=40 \\\n"
        f"#       diffusion_rollout=True \\\n"
        f"#       fe_diffusion_num_ensemble_members=1 \\\n"
        f"#       data_loading.num_workers=0\n"
        f"#\n"
        f"# If the model run was decoder-finetuned at a narrow noise band (sigma_max close to\n"
        f"# sigma_min), add sigma_max=<parent's value> to --options: diffusion.py caps the ODE at\n"
        f"# min(cf.sigma_max, sigma_max_train), so the dcft band would start the rollout already\n"
        f"# almost denoised.\n"
    )
    with out.open("w") as fh:
        fh.write(header)
        yaml.safe_dump(overlay, fh, sort_keys=False, width=100)

    print(f"wrote {out}")
    print(f"  decoded stream : {stream}")
    print(f"  not reconstructed: {[k for k in streams if k != stream] or '(none)'}")
    print(f"\nuse with:\n  uv run inference --from-run-id {args.model} --config {out} --options ...")


if __name__ == "__main__":
    main()
