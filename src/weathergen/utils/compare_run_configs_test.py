import json
import sys
from pathlib import Path

import pytest

from weathergen.utils.compare_run_configs import main


def _make_run(models_parent: Path, run_id: str) -> Path:
    """Create a minimal run directory with a model config inside."""
    run_dir = models_parent / run_id
    run_dir.mkdir(parents=True)
    config = {
        "general": {"run_id": run_id},
        "streams": {"ERA5": {"name": "ERA5"}},
        "training_config": {"learning_rate": 0.001},
    }
    (run_dir / f"model_{run_id}.json").write_text(json.dumps(config))
    return run_dir


def _run_compare(tmp_path: Path, entries: list, monkeypatch: pytest.MonkeyPatch) -> str:
    config_list = tmp_path / "compare_config_list.yml"
    config_list.write_text(
        "run_ids:\n"
        + "".join(f"  - {entry!r}\n" for entry in entries)
        + "always_show_patterns:\n  - '*'\n",
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "compare_configs.md"
    monkeypatch.setattr(
        sys, "argv", ["compare_run_configs.py", "--config", str(config_list), str(output)]
    )
    main()
    # The tool appends the compared run ids to the requested output name.
    reports = sorted((tmp_path / "reports").glob("*.md"))
    assert reports, "the tool did not write a markdown report"
    return reports[0].read_text(encoding="utf-8")


def test_bare_path_entry_loads_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A bare YAML entry is the run directory; its name is the run_id.

    Since #1704 load_run_config appends run_id to model_path, so the tool must
    pass the entry's parent directory — otherwise it looks for the config in a
    doubled <run_dir>/<run_id> path and crashes.
    """
    run_dir = _make_run(tmp_path / "models", "myrun")

    table = _run_compare(tmp_path, [str(run_dir)], monkeypatch)

    assert "myrun" in table


def test_run_id_and_directory_entry_loads_from_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The documented [run_id, model_parent_dir] format keeps working."""
    models_parent = tmp_path / "models"
    _make_run(models_parent, "myrun")

    table = _run_compare(tmp_path, [["myrun", str(models_parent)]], monkeypatch)

    assert "myrun" in table
