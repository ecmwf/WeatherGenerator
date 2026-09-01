"""
Performance integration test for the Weather Generator (issue #2092).

Runs a short training of the default config (shortened by
integration_tests/performance.yaml) and asserts that the global training
throughput is above a threshold.

The test must run on a GPU node, with one task per GPU, e.g. via:
    sbatch integration_tests/performance_slurm.sh
"""

import logging
import os
import shutil
from pathlib import Path

import pytest

from weathergen.run_train import main
from weathergen.utils.metrics import get_train_metrics_path, read_metrics_file

logger = logging.getLogger(__name__)

# Read from git the current commit hash and take the first 5 characters:
try:
    from git import Repo

    repo = Repo(search_parent_directories=False)
    commit_hash = repo.head.object.hexsha[:5]
    logger.info(f"Current commit hash: {commit_hash}")
except Exception as e:
    commit_hash = "unknown"
    logger.warning(f"Could not get commit hash: {e}")

WEATHERGEN_HOME = Path(__file__).parent.parent

THROUGHPUT_METRIC = "performance.throughput.global.samples_per_sec"

# Measured baselines on one Santis node (4x GH200): 2.46-2.97 global
# samples/sec across runs; the threshold leaves ~20% margin below the lowest
# observed value. Revisit when the config or the node type changes.
THROUGHPUT_THRESHOLD = 2.0


def is_rank_zero() -> bool:
    """Whether this process is rank 0.

    Uses SLURM_PROCID rather than torch.distributed state: it is set by SLURM
    before Python starts, so it is reliable even if DDP initialization or
    training crashed on some rank.
    """
    return os.environ.get("SLURM_PROCID", "0") == "0"


@pytest.fixture()
def setup(test_run_id):
    logger.info(f"setup fixture with {test_run_id}")
    if is_rank_zero():
        shutil.rmtree(WEATHERGEN_HOME / "results" / test_run_id, ignore_errors=True)
        shutil.rmtree(WEATHERGEN_HOME / "models" / test_run_id, ignore_errors=True)
    yield
    logger.info("end fixture")


# Include the SLURM job id so concurrent or back-to-back jobs from the same
# commit cannot delete or overwrite each other's results.
@pytest.mark.parametrize(
    "test_run_id", ["test_perf_" + commit_hash + "_" + os.environ.get("SLURM_JOB_ID", "local")]
)
def test_performance(setup, test_run_id):
    logger.info(f"test_performance with run_id {test_run_id} {WEATHERGEN_HOME}")

    main(
        [
            "train",
            "--config",
            f"{WEATHERGEN_HOME}/integration_tests/performance.yaml",
            "--run-id",
            test_run_id,
        ]
    )

    # Only rank 0 writes the metrics file, so only rank 0 can assert on it.
    if is_rank_zero():
        assert_throughput(test_run_id)
    logger.info("end test_performance")


def assert_throughput(run_id):
    """Assert that the final global training throughput is above the threshold."""
    throughput = load_final_throughput(run_id)
    logger.info(f"{THROUGHPUT_METRIC} = {throughput:.3f} (threshold {THROUGHPUT_THRESHOLD})")
    assert throughput >= THROUGHPUT_THRESHOLD, (
        f"Throughput regression: {THROUGHPUT_METRIC} = {throughput:.3f} samples/sec "
        f"is below the threshold of {THROUGHPUT_THRESHOLD} samples/sec."
    )


def load_final_throughput(run_id) -> float:
    """Load the last logged value of the global throughput metric.

    The tracker reports cumulative throughput since warmup, so the last value
    averages over the longest window and is the most stable.
    """
    # Mirrors how train_logger resolves the path: the writer passes the run
    # directory (get_path_run) as base_path. WEATHERGEN_HOME/results must be a
    # symlink to the shared working directory (scripts/actions.sh create-links).
    metrics_path = get_train_metrics_path(
        base_path=WEATHERGEN_HOME / "results" / run_id, run_id=run_id
    )
    # run_train.main swallows exceptions when world_size > 1, so a missing
    # metrics file is the signal that training crashed.
    assert metrics_path.exists(), (
        f"Metrics file not found for run_id {run_id}: {metrics_path}. "
        "Training most likely crashed; check the training logs."
    )
    metrics = read_metrics_file(metrics_path)
    assert THROUGHPUT_METRIC in metrics.columns, (
        f"{THROUGHPUT_METRIC} not found in {metrics_path}. "
        "Is train_logging.track_performance_metrics enabled?"
    )
    values = metrics[THROUGHPUT_METRIC].drop_nulls()
    values = values.filter(values.is_not_nan())
    assert len(values) > 0, f"No values logged for {THROUGHPUT_METRIC} in {metrics_path}"
    return float(values.tail(1).item())
