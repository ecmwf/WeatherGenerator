from io import StringIO
from math import isnan
from pathlib import Path

from weathergen.utils.metrics import (
    get_train_metrics_path,
    read_metrics_file,
)

s = """{"weathergen.timestamp":100, "m": "nan"}
{"weathergen.timestamp":101,"m": 1.3}
{"weathergen.timestamp":102,"a": 4}
"""


def test1():
    df = read_metrics_file(StringIO(s))
    assert df.shape == (3, 3)
    assert df["weathergen.timestamp"].to_list() == [100, 101, 102]
    assert isnan(df["m"].to_list()[0])
    assert df["m"].to_list()[1:] == [1.3, None]
    assert df["a"].to_list() == [None, None, 4]


def _write_legacy_metrics(run_dir: Path) -> Path:
    # Layout used before the metrics file was renamed (#590):
    # results/{RUN-ID}/metrics.json
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.json"
    metrics.write_text('{"weathergen.timestamp":100, "loss_avg": 1.0}\n')
    return metrics


def test_train_metrics_path_finds_legacy_file_next_to_run_outputs(tmp_path: Path):
    """Callers pass base_path = results/{RUN-ID}; the legacy file lives in it."""
    run_id = "legacy_run"
    run_dir = tmp_path / run_id
    legacy_file = _write_legacy_metrics(run_dir)

    metrics_path = get_train_metrics_path(run_dir, run_id)

    assert metrics_path == legacy_file
    df = read_metrics_file(metrics_path)
    assert df["loss_avg"].to_list() == [1.0]


def test_train_metrics_path_finds_legacy_file_from_results_parent(tmp_path: Path):
    """Original #601 callers passed the results parent as base_path."""
    run_id = "legacy_run"
    legacy_file = _write_legacy_metrics(tmp_path / run_id)

    metrics_path = get_train_metrics_path(tmp_path, run_id)

    assert metrics_path == legacy_file
    df = read_metrics_file(metrics_path)
    assert df["loss_avg"].to_list() == [1.0]
