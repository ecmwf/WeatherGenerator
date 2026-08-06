"""Normalise legacy boolean plot flags to list-based ``data_plots``/``score_plots`` config.

Detects old-style booleans, converts to lists, and emits DeprecationWarning.
"""

from __future__ import annotations

import logging
import warnings

_logger = logging.getLogger(__name__)

# ── Supported values ─────────────────────────────────────────────────────────

SUPPORTED_DATA_PLOTS = frozenset(
    {"maps", "bias", "target", "histograms", "animations", "timeseries"}
)
SUPPORTED_SCORE_PLOTS = frozenset(
    {
        "lead_time",
        "ratio",
        "heatmap",
        "scorecard",
        "bar",
        "qq",
        "rank_histogram",
        "score_map",
        "score_animation",
        "timeseries",
    }
)

# ── Old boolean key → new list entry ─────────────────────────────────────────

_DATA_PLOT_BOOL_MAP = {
    "plot_maps": "maps",
    "plot_bias": "bias",
    "plot_target": "target",
    "plot_histograms": "histograms",
    "plot_animations": "animations",
    "plot_timeseries": "timeseries",
}
_SCORE_PLOT_BOOL_MAP = {
    "summary_plots": "lead_time",
    "ratio_plots": "ratio",
    "heat_maps": "heatmap",
    "score_cards": "scorecard",
    "bar_plots": "bar",
    "plot_score_maps": "score_map",
    "plot_score_animations": "score_animation",
    "plot_score_init_timeseries": "timeseries",
}


# ── Public API ───────────────────────────────────────────────────────────────


def parse_data_plots(plotting_cfg: dict | None) -> list[str]:
    """Convert per-stream plotting config to a validated ``data_plots`` list."""
    if not plotting_cfg:
        return []
    if "data_plots" in plotting_cfg:
        result = list(plotting_cfg["data_plots"])
        _validate(result, SUPPORTED_DATA_PLOTS, "data_plots")
        return result
    return _convert_bools(
        plotting_cfg,
        _DATA_PLOT_BOOL_MAP,
        "data_plots",
        SUPPORTED_DATA_PLOTS,
        histograms_special=True,
    )


def parse_score_plots(eval_cfg: dict | None) -> list[str]:
    """Convert evaluation config to a validated ``score_plots`` list."""
    if not eval_cfg:
        return []
    if "score_plots" in eval_cfg:
        result = list(eval_cfg["score_plots"])
        _validate(result, SUPPORTED_SCORE_PLOTS, "score_plots")
        return result
    return _convert_bools(eval_cfg, _SCORE_PLOT_BOOL_MAP, "score_plots", SUPPORTED_SCORE_PLOTS)


def parse_plot_config(cfg: dict) -> dict:
    """Normalise full config in-place: resolve ``score_plots`` + per-stream ``data_plots``."""
    eval_cfg = cfg.get("evaluation") or {}
    _set_key(eval_cfg, "score_plots", parse_score_plots(eval_cfg))

    for stream_cfg in (cfg.get("default_streams") or {}).values():
        if isinstance(stream_cfg, dict) and stream_cfg.get("plotting") is not None:
            _set_key(stream_cfg["plotting"], "data_plots", parse_data_plots(stream_cfg["plotting"]))

    for run_cfg in (cfg.get("run_ids") or {}).values():
        if not isinstance(run_cfg, dict):
            continue
        for stream_cfg in (run_cfg.get("streams") or {}).values():
            if isinstance(stream_cfg, dict) and stream_cfg.get("plotting") is not None:
                _set_key(
                    stream_cfg["plotting"], "data_plots", parse_data_plots(stream_cfg["plotting"])
                )
    return cfg


def get_plot_score_options(eval_cfg: dict) -> dict[str, bool]:
    """Bridge: derive legacy ``plot_score_options`` dict from ``score_plots`` list."""
    sp = set(eval_cfg.get("score_plots", []))
    return {
        "plot_score_maps": "score_map" in sp,
        "plot_score_animations": "score_animation" in sp,
        "plot_score_init_time_series": "timeseries" in sp,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _convert_bools(cfg, bool_map, field_name, supported, *, histograms_special=False):
    """Convert old-style boolean flags to a list, emitting a deprecation warning."""
    result, found = [], False
    for old_key, new_entry in bool_map.items():
        value = cfg.get(old_key)
        if value is None:
            continue
        found = True
        if histograms_special and old_key == "plot_histograms":
            if value is True or value in ("per-sample", "across-samples"):
                result.append(new_entry)
        elif value:
            result.append(new_entry)
    if found:
        warnings.warn(
            f"Boolean plot flags are deprecated. Use '{field_name}: {result}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return result


def _validate(values, supported, field_name):
    unknown = set(values) - supported
    if unknown:
        raise ValueError(
            f"Unsupported values in '{field_name}': {sorted(unknown)}. "
            f"Supported: {sorted(supported)}"
        )


def _set_key(cfg, key, value):
    """Set a key on a dict or OmegaConf DictConfig."""
    try:
        cfg[key] = value
    except Exception:
        try:
            setattr(cfg, key, value)  # pylint: disable=bad-builtin
        except Exception:
            _logger.debug(f"Could not set '{key}' on {type(cfg)}")
