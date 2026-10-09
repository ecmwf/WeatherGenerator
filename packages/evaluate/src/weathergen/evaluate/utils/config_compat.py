"""Parse the plot options of the evaluation config.

* ``evaluation.score_plots``: list of score visualisations.  Legacy boolean flags are
  converted with a DeprecationWarning.
* ``<stream>.plotting.data_plots``: list of data visualisations, each produced as images
  and/or videos (see :func:`parse_data_plots`).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

from omegaconf import DictConfig, ListConfig

_logger = logging.getLogger(__name__)

# ── Supported values ─────────────────────────────────────────────────────────

MAP_KINDS = ("predictions", "target", "bias")
PLOT_FORMATS = ("image", "video")
# data_plots histogram entries -> histogram kinds they produce
HISTOGRAM_ENTRIES = {
    "histograms": ("per_sample", "across_samples"),
    "histograms_per_sample": ("per_sample",),
    "histograms_across_samples": ("across_samples",),
}
SUPPORTED_DATA_PLOTS = ("maps", *HISTOGRAM_ENTRIES, "timeseries")
SUPPORTED_SCORE_PLOTS = frozenset(
    {
        "metric_plots",  # standard plot of each metric (line, Q-Q or PSD plot)
        "ratio",
        "heatmap",
        "scorecard",
        "bar",
        "score_map",
        "score_animation",
        "init_hour",  # score vs initialisation hour of the day
    }
)

# Renamed score_plots values.
_DEPRECATED_SCORE_PLOTS = {"lead_time": "metric_plots"}

_DATA_PLOTS_EXAMPLE = """\
  data_plots:
    - maps:
        predictions: [image, video]
        target: [image]
        bias: [image, video]
    - histograms_per_sample        # or histograms_per_sample: [image, video]
    - timeseries"""

# Plotting keys of the former syntax, rejected with a pointer to data_plots.
_LEGACY_DATA_PLOT_KEYS = (
    "plot_maps",
    "plot_bias",
    "plot_target",
    "plot_histograms",
    "plot_animations",
    "plot_timeseries",
)

# ── Old boolean key → new list entry ─────────────────────────────────────────

_SCORE_PLOT_BOOL_MAP = {
    "summary_plots": "metric_plots",
    "ratio_plots": "ratio",
    "heat_maps": "heatmap",
    "score_cards": "scorecard",
    "bar_plots": "bar",
    "plot_score_maps": "score_map",
    "plot_score_animations": "score_animation",
    "plot_score_init_timeseries": "init_hour",
    "plot_score_init_time_series": "init_hour",  # key read before the list-based config
}


# ── Public API ───────────────────────────────────────────────────────────────


@dataclass
class DataPlotSpec:
    """Data plots requested for one stream, each with its output formats.

    A format set contains ``"image"`` and/or ``"video"``.  Videos are animations over
    forecast steps built from the per-step images, so those images are always written.
    """

    maps: dict[str, frozenset[str]] = field(default_factory=dict)  # map kind -> formats
    histograms: dict[str, frozenset[str]] = field(default_factory=dict)  # hist kind -> formats
    timeseries: bool = False

    def __bool__(self) -> bool:
        return bool(self.maps or self.histograms or self.timeseries)

    def map_videos(self) -> set[str]:
        """Map kinds to animate."""
        return {kind for kind, formats in self.maps.items() if "video" in formats}

    def histogram_videos(self) -> set[str]:
        """Histogram kinds (``per_sample``/``across_samples``) to animate."""
        return {kind for kind, formats in self.histograms.items() if "video" in formats}


def parse_data_plots(plotting_cfg: Mapping | None) -> DataPlotSpec:
    """Parse ``plotting.data_plots`` of one stream into a :class:`DataPlotSpec`.

    Each list entry is either a plain name or a single-key mapping::

        data_plots:
          - maps:                          # map kind -> formats
              predictions: [image, video]
              target: [image]
              bias: [video]
          - histograms_per_sample          # images; or histograms_per_sample: [image, video]
          - timeseries

    ``maps`` alone means ``predictions: [image]``; ``histograms`` covers both
    ``histograms_per_sample`` and ``histograms_across_samples``.
    """
    spec = DataPlotSpec()
    if not plotting_cfg:
        return spec

    legacy = [key for key in _LEGACY_DATA_PLOT_KEYS if key in plotting_cfg]
    if legacy:
        raise ValueError(
            f"Plotting options {legacy} are no longer supported. "
            f"Use 'data_plots' instead, e.g.:\n{_DATA_PLOTS_EXAMPLE}"
        )

    entries = plotting_cfg.get("data_plots")
    if entries is None:
        return spec
    if not isinstance(entries, list | ListConfig):
        raise ValueError(f"'data_plots' must be a list, e.g.:\n{_DATA_PLOTS_EXAMPLE}")

    for entry in entries:
        name, options = _split_entry(entry)
        if name == "maps":
            kinds = {"predictions": ["image"]} if options is None else options
            if not isinstance(kinds, Mapping):
                raise ValueError(
                    "'maps' takes a mapping of map kind to formats "
                    f"(kinds: {list(MAP_KINDS)}), e.g.:\n{_DATA_PLOTS_EXAMPLE}"
                )
            for kind, formats in kinds.items():
                if kind not in MAP_KINDS:
                    raise ValueError(
                        f"Unsupported map kind '{kind}' in 'data_plots'. "
                        f"Supported: {list(MAP_KINDS)}"
                    )
                spec.maps[kind] = spec.maps.get(kind, frozenset()) | _parse_formats(
                    formats, f"maps.{kind}"
                )
        elif name in HISTOGRAM_ENTRIES:
            formats = _parse_formats(["image"] if options is None else options, name)
            for kind in HISTOGRAM_ENTRIES[name]:
                spec.histograms[kind] = spec.histograms.get(kind, frozenset()) | formats
        elif name == "timeseries":
            if options is not None:
                raise ValueError("'timeseries' in 'data_plots' takes no options.")
            spec.timeseries = True
        else:
            raise ValueError(
                f"Unsupported entry '{name}' in 'data_plots'. "
                f"Supported: {list(SUPPORTED_DATA_PLOTS)}, e.g.:\n{_DATA_PLOTS_EXAMPLE}"
            )
    return spec


def parse_score_plots(eval_cfg: dict | None) -> list[str]:
    """Convert evaluation config to a validated ``score_plots`` list."""
    if not eval_cfg:
        return []
    if "score_plots" in eval_cfg:
        result = _replace_deprecated_score_plots(list(eval_cfg["score_plots"]))
        _validate(result, SUPPORTED_SCORE_PLOTS, "score_plots")
        return result
    return _convert_bools(eval_cfg, _SCORE_PLOT_BOOL_MAP, "score_plots", SUPPORTED_SCORE_PLOTS)


def parse_plot_config(cfg: dict) -> dict:
    """Resolve ``score_plots`` in place and validate every stream's ``data_plots``.

    ``data_plots`` is validated up front so that config errors surface before any data is
    loaded; it is parsed again where the plots are made.
    """
    eval_cfg = cfg.get("evaluation") or {}
    _set_key(eval_cfg, "score_plots", parse_score_plots(eval_cfg))

    # The config is usually an OmegaConf DictConfig, which is not a ``dict``.
    stream_cfgs = list((cfg.get("default_streams") or {}).values())
    for run_cfg in (cfg.get("run_ids") or {}).values():
        if isinstance(run_cfg, dict | DictConfig):
            stream_cfgs.extend((run_cfg.get("streams") or {}).values())
    for stream_cfg in stream_cfgs:
        if isinstance(stream_cfg, dict | DictConfig) and stream_cfg.get("plotting") is not None:
            parse_data_plots(stream_cfg["plotting"])
    return cfg


def get_plot_score_options(eval_cfg: dict) -> dict[str, bool]:
    """Bridge: derive legacy ``plot_score_options`` dict from ``score_plots`` list."""
    sp = set(eval_cfg.get("score_plots", []))
    return {
        "plot_score_maps": "score_map" in sp,
        "plot_score_animations": "score_animation" in sp,
        "plot_score_init_time_series": "init_hour" in sp,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _split_entry(entry) -> tuple[str, object]:
    """Return ``(name, options)`` of a ``data_plots`` entry; options is None for a plain name."""
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, Mapping) and len(entry) == 1:
        return next(iter(entry.items()))
    raise ValueError(
        f"Each 'data_plots' entry must be a name or a single-key mapping, got {entry!r}. "
        f"Example:\n{_DATA_PLOTS_EXAMPLE}"
    )


def _parse_formats(formats, where: str) -> frozenset[str]:
    """Validate a list of output formats (``image``/``video``)."""
    if isinstance(formats, str) or not isinstance(formats, list | ListConfig) or not formats:
        raise ValueError(
            f"'{where}' in 'data_plots' needs a non-empty list of formats from "
            f"{list(PLOT_FORMATS)}, got {formats!r}."
        )
    unknown = set(formats) - set(PLOT_FORMATS)
    if unknown:
        raise ValueError(
            f"Unsupported format(s) {sorted(unknown)} for '{where}' in 'data_plots'. "
            f"Supported: {list(PLOT_FORMATS)}"
        )
    return frozenset(formats)


def _convert_bools(cfg, bool_map, field_name, supported):
    """Convert old-style boolean flags to a list, emitting a deprecation warning."""
    result, found = [], False
    for old_key, new_entry in bool_map.items():
        value = cfg.get(old_key)
        if value is None:
            continue
        found = True
        if value:
            result.append(new_entry)
    if found:
        warnings.warn(
            f"Boolean plot flags are deprecated. Use '{field_name}: {result}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return result


def _replace_deprecated_score_plots(values: list[str]) -> list[str]:
    """Map renamed ``score_plots`` values to their replacement, with a deprecation warning."""
    result = []
    for value in values:
        new = _DEPRECATED_SCORE_PLOTS.get(value, value)
        if new != value:
            warnings.warn(
                f"score_plots value '{value}' is deprecated. Use '{new}' instead.",
                DeprecationWarning,
                stacklevel=4,
            )
        if new not in result:
            result.append(new)
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
