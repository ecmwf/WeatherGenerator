from pathlib import Path

from weathergen.utils.metrics import read_metrics_file
from weathergen.utils.train_logger import TrainLogger


def test_log_metrics_writes_run_metrics_file_without_config(tmp_path: Path):
    """TrainLogger needs only the run_id and the results path.

    The constructor must not require the global config object: the metrics
    file location is fully determined by (path_results, run_id).
    """
    run_id = "test_run_123"
    logger = TrainLogger(run_id, tmp_path)

    logger.log_metrics("train", {"loss_avg": 1.5}, step=3)

    metrics_path = tmp_path / f"{run_id}_train_metrics.json"
    assert metrics_path.exists()
    df = read_metrics_file(metrics_path)
    assert df["loss_avg"].to_list() == [1.5]
    assert df["weathergen.step"].to_list() == [3]
    assert df["stage"].to_list() == ["train"]
