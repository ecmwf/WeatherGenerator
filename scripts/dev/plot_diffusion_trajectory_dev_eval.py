# ruff: noqa: T201

"""Evaluate/plot the diffusion denoising trajectory written in dev mode with the standard
``evaluate`` (fast-eval) package.

When ``fe_diffusion_dev_mode: true`` is set, ``weathergen.utils.validation_io`` writes one
synthetic "forecast step" per ODE denoising step to
``<run>/dev/diffusion_trajectory/validation_chkpt{mini_epoch:05d}_rank{rank:04d}.zarr`` (see
``write_dev_trajectory_output``). The file uses the exact same schema as the official
validation/test output, so it can be scored and plotted with the unmodified ``evaluate``
package by simply pointing ``results_base_dir`` at the dev directory instead of the run's
normal results directory. The "forecast_step" axis in the resulting plots corresponds to the
ODE denoising step, not a physical lead time.

Usage:
    uv run python scripts/dev/plot_diffusion_trajectory_dev_eval.py --run-id <id> --mini-epoch 0
    uv run python scripts/dev/plot_diffusion_trajectory_dev_eval.py --run-id <id> \\
        --config config/evaluate/eval_config_diffusion.yml
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

import weathergen.common.config as config
from weathergen.evaluate.run_evaluation import evaluate_from_config
from weathergen.utils.validation_io import get_dev_trajectory_dir

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_CONFIG = (
    Path(__file__).parent.parent.parent / "config" / "evaluate" / "eval_config_diffusion.yml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run id the dev trajectory was written for.")
    parser.add_argument("--mini-epoch", type=int, default=0, help="Checkpoint mini_epoch to read.")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_EVAL_CONFIG),
        help="Base evaluate config providing metrics/plotting options (run_ids is overwritten).",
    )
    parser.add_argument(
        "--rank", default="all", help="Rank file(s) to read: an int, 'all', or a list of ints."
    )
    args = parser.parse_args()

    cf = config.load_run_config(run_id=args.run_id, mini_epoch=args.mini_epoch, model_path=None)
    dev_dir = get_dev_trajectory_dir(cf)
    if not dev_dir.exists():
        raise FileNotFoundError(
            f"No dev trajectory output found at {dev_dir}. Was the run started with "
            "fe_diffusion_dev_mode=true?"
        )

    eval_cf = OmegaConf.load(args.config)
    OmegaConf.set_struct(eval_cf, False)
    eval_cf.run_ids = {
        args.run_id: {
            "label": f"{args.run_id} (diffusion dev trajectory, ODE step as forecast_step)",
            "results_base_dir": str(dev_dir),
            "mini_epoch": args.mini_epoch,
            "rank": args.rank,
        }
    }

    logger.info("Evaluating dev trajectory for run %s from %s", args.run_id, dev_dir)
    evaluate_from_config(eval_cf, mlflow_client=None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
