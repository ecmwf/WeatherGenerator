# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

# Third-party
import numpy as np
import omegaconf as oc
import xarray as xr
import zarr
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

# Local application / package
from weathergen.common.config import (
    get_path_run,
    load_merge_configs,
    load_run_config,
)
from weathergen.common.io import zarrio_reader
from weathergen.evaluate.io.io_reader import Reader, ReaderOutput
from weathergen.evaluate.scores.score_utils import to_list
from weathergen.evaluate.utils.derived_channels import DeriveChannels

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def _process_sample_result(
    out,
    sample: int,
    fstep: int,
    ensemble: list[str],
    is_gridded: bool,
) -> tuple[int, int, xr.DataArray, xr.DataArray, list | None] | None:
    """Post-process a single ZarrIO.get_data() result.

    Shared logic used by both _load_single_sample (shared context) and
    _load_single_sample_own_context (per-thread context).
    """
    if out.target is None or out.prediction is None:
        return None

    target = out.target.as_xarray()
    pred = out.prediction.as_xarray()

    if len(target.ipoint) == 0:
        return None

    if ensemble == ["mean"]:
        pred = pred.mean("ens", keepdims=True)
    else:
        pred = pred.sel(ens=ensemble)

    pred = pred.squeeze()
    target = target.squeeze()

    # Force materialisation inside this thread — .load() converts dask→numpy
    # while keeping xarray metadata (dims, coords, attrs).  This is the key
    # difference vs .persist() which only schedules the compute and can end
    # up serialising I/O on the main-thread dask scheduler.
    target = target.load()
    pred = pred.load()

    valid_times = None
    if is_gridded:
        vt_list = np.unique(target.valid_time.values).tolist()
        if len(vt_list) > 1:
            valid_times = vt_list

    return fstep, sample, target, pred, valid_times


def _load_single_sample(
    zio,
    sample: int,
    stream: str,
    fstep: int,
    ensemble: list[str],
    is_gridded: bool,
) -> tuple[int, int, xr.DataArray, xr.DataArray, list | None] | None:
    """
    Load and preprocess one (sample, fstep) from a *shared* ZarrIO context.

    Thread-safe for LocalStore backends. The zio object's data_root
    (a zarr.Group) supports concurrent read-only path lookups.
    NOT safe for ZipStore — use _load_single_sample_own_context instead.

    Returns (fstep, sample_idx, target, prediction, valid_times) or None
    if empty/missing.
    """
    out = zio.get_data(sample, stream, fstep)
    return _process_sample_result(out, sample, fstep, ensemble, is_gridded)


def _load_single_sample_own_context(
    fname_zarr: Path,
    sample: int,
    stream: str,
    fstep: int,
    ensemble: list[str],
    is_gridded: bool,
) -> tuple[int, int, xr.DataArray, xr.DataArray, list | None] | None:
    """
    Load and preprocess one (sample, fstep) using the **worker-global** ZarrIO.

    Used with ProcessPoolExecutor for ZipStore backends.  Each worker
    process keeps a single persistent ZarrIO handle (initialised by
    ``_init_worker_zio``) so the zip central directory is parsed only
    once per process, not once per item.

    Returns (fstep, sample_idx, target, prediction, valid_times) or None
    if empty/missing.
    """
    global _worker_zio  # noqa: PLW0602
    if _worker_zio is None:
        # Lazy open — should not normally happen if initializer ran
        _worker_zio = zarrio_reader(fname_zarr).__enter__()
    out = _worker_zio.get_data(sample, stream, fstep)
    return _process_sample_result(out, sample, fstep, ensemble, is_gridded)


# Worker-global ZarrIO handle for ProcessPoolExecutor workers
_worker_zio = None


def _init_worker_zio(fname_zarr: Path) -> None:
    """ProcessPoolExecutor initializer: open a persistent ZarrIO per worker."""
    global _worker_zio  # noqa: PLW0603
    _worker_zio = zarrio_reader(fname_zarr).__enter__()


def _compute_early_channel_selection(
    read_channels: list[str],
    requested_channels: list[str],
    stream_cfg: dict,
) -> tuple[list[int] | None, list[str]]:
    """Compute channel indices for early selection in _read_sample_raw.

    When the stream does NOT use derived channels, we can select only the
    requested channels at the numpy level inside each worker, avoiding
    the transfer and stacking of unrequested channels.

    Parameters
    ----------
    read_channels : list[str]
        Full list of channels available in the zarr store.
    requested_channels : list[str]
        Channels the user requested for evaluation/plotting.
    stream_cfg : dict
        Stream configuration (checked for ``derive_channels`` key).

    Returns
    -------
    channel_idxs : list[int] | None
        Indices into ``read_channels`` to select, or ``None`` to read all.
    effective_channels : list[str]
        Channel names that will be returned (subset or full list).
    """
    # If derived channels are configured, we must keep ALL channels so that
    # the derivation logic in _select_channels has its source data.
    if "derive_channels" in stream_cfg:
        return None, read_channels

    # Find the intersection: requested channels that exist in the zarr store
    available_set = set(read_channels)
    needed = [ch for ch in requested_channels if ch in available_set]

    # If all channels are requested anyway, or the intersection is empty,
    # skip early selection.
    if not needed or len(needed) == len(read_channels):
        return None, read_channels

    # Build index list preserving zarr order for stable indexing
    chan_to_idx = {ch: i for i, ch in enumerate(read_channels)}
    idxs = sorted(chan_to_idx[ch] for ch in needed)
    effective = [read_channels[i] for i in idxs]

    _logger.debug(
        f"Early channel selection: {len(effective)}/{len(read_channels)} channels "
        f"({', '.join(effective[:5])}{'...' if len(effective) > 5 else ''})"
    )
    return idxs, effective


def _read_sample_raw(
    zarr_path: str,
    sample: int,
    stream: str,
    fsteps: list[int],
    channel_idxs: list[int],
    is_zip: bool,
    read_coords: bool = False,
    is_gridded: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """
    Read all forecast steps for one sample via direct zarr array access.

    Bypasses ZarrIO / OutputDataset / as_xarray / dask for maximum speed.
    Each worker opens its own zarr store handle (safe for both ZipStore and
    LocalStore).

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store (.zarr directory or .zip file).
    sample : int
        Sample index to read.
    stream : str
        Stream name (e.g. "ERA5").
    fsteps : list[int]
        Forecast steps to read.
    channel_idxs : list[int]
        Pre-computed indices into the channel axis (select only needed channels).
    is_zip : bool
        Whether the store is a ZipStore (.zip).
    read_coords : bool
        If True, also read per-sample coords (needed for scatter/non-gridded data).
    is_gridded : bool
        If True, split by unique valid_times to create sub-steps (gridded data
        with multiple forecast sub-steps per fstep).  If False (scatter/obs data),
        keep all observations in a single array per fstep — each observation has
        its own time and splitting would create one array per observation.

    Returns
    -------
    preds_all : list[np.ndarray]
        Per-fstep prediction arrays, shape (ipoints, channels[, ens]).
        For sub-steps, multiple arrays per fstep (one per unique valid_time).
    targets_all : list[np.ndarray]
        Per-fstep target arrays, shape (ipoints, channels).
    times_all : list[np.ndarray]
        Per-fstep time arrays (unique times per entry).
    meta : dict
        Metadata: {"source_interval": ..., "n_substeps": list[int],
                    "coords": np.ndarray | None}.
    """
    if is_zip:
        store = zarr.storage.ZipStore(zarr_path, mode="r")
        ds = zarr.open_group(store=store, mode="r")
    else:
        store = zarr.storage.LocalStore(zarr_path)
        ds = zarr.open_group(store=store, mode="r")

    preds_all, targets_all, times_all = [], [], []
    n_substeps = []  # track how many sub-steps per fstep
    source_interval = None

    for fs in fsteps:
        base = f"{sample}/{stream}/{fs}"

        # Read source_interval from prediction group attributes (once)
        if source_interval is None:
            try:
                pred_group = ds[f"{base}/prediction"]
                attrs = dict(pred_group.attrs)
                source_interval = attrs.get("source_interval", {})
            except (KeyError, AttributeError):
                source_interval = {}

        # Direct array access — bypasses OutputDataset/as_xarray/dask entirely
        pred_data = np.asarray(ds[f"{base}/prediction/data"])
        target_data = np.asarray(ds[f"{base}/target/data"])
        times_data = np.asarray(ds[f"{base}/prediction/times"])

        # Select channels by index
        if channel_idxs is not None:
            pred_data = (
                pred_data[:, channel_idxs] if pred_data.ndim == 2 else pred_data[:, channel_idxs, :]
            )
            target_data = target_data[:, channel_idxs]

        # Handle sub-steps (gridded data with multiple valid_times per fstep).
        # For scatter/observation data each observation has its own timestamp,
        # so splitting by unique time would create one tiny array per obs —
        # thousands of them — causing the assembly code to hang.
        unique_times = np.unique(times_data)
        if is_gridded and len(unique_times) > 1:
            count = 0
            for ut in unique_times:
                mask = times_data == ut
                preds_all.append(pred_data[mask])
                targets_all.append(target_data[mask])
                count += 1
            times_all.append(unique_times)
            n_substeps.append(count)
        else:
            preds_all.append(pred_data)
            targets_all.append(target_data)
            # For scatter data, keep the full per-observation times array
            # so the DataArray builder can assign per-ipoint valid_time.
            # For gridded data with 1 unique time, unique_times suffices.
            if not is_gridded:
                times_all.append(times_data)
            else:
                times_all.append(unique_times)
            n_substeps.append(1)

    # Optionally read per-sample coordinates (for scatter / non-gridded data)
    sample_coords = None
    if read_coords and fsteps:
        try:
            base0 = f"{sample}/{stream}/{fsteps[0]}"
            sample_coords = np.asarray(ds[f"{base0}/prediction/coords"])
        except (KeyError, AttributeError):
            pass

    try:
        store.close()
    except Exception:
        pass

    meta = {
        "source_interval": source_interval,
        "n_substeps": n_substeps,
        "coords": sample_coords,
    }
    return preds_all, targets_all, times_all, meta


def _read_coords_and_meta(
    zarr_path: str,
    stream: str,
    fstep: int,
    is_zip: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Read coordinates and channel names from the zarr store (once).

    Returns
    -------
    coords : np.ndarray, shape (ipoints, 2) — lat, lon
    channels : list[str] — all channel names from zarr
    times_ref : np.ndarray — reference times from sample 0
    """
    if is_zip:
        store = zarr.storage.ZipStore(zarr_path, mode="r")
        ds = zarr.open_group(store=store, mode="r")
    else:
        store = zarr.storage.LocalStore(zarr_path)
        ds = zarr.open_group(store=store, mode="r")

    base = f"0/{stream}/{fstep}"
    coords = np.asarray(ds[f"{base}/prediction/coords"])
    times_ref = np.asarray(ds[f"{base}/prediction/times"])

    # Read channel names from group attributes
    pred_group = ds[f"{base}/prediction"]
    channels = list(pred_group.attrs.get("channels", []))

    try:
        store.close()
    except Exception:
        pass

    return coords, channels, times_ref


class WeatherGenReader(Reader):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        super().__init__(eval_cfg, run_id, private_paths)

        # TODO: remove backwards compatibility to "epoch" in Feb. 2026
        self.mini_epoch = eval_cfg.get("mini_epoch", 0)
        self.rank = eval_cfg.get("rank", 0)

        # Load model configuration and set (run-id specific) directories
        self.inference_cfg = self.get_inference_config()

        if not self.results_base_dir:
            self.results_base_dir = get_path_run(self.inference_cfg)
            _logger.info(f"Results directory obtained from private config: {self.results_base_dir}")
        else:
            _logger.info(f"Results directory parsed: {self.results_base_dir}")

        self.runplot_base_dir = Path(
            self.eval_cfg.get("runplot_base_dir", self.results_base_dir)
        )  # base directory where map plots and histograms will be stored

        self.metrics_base_dir = Path(
            self.eval_cfg.get("metrics_base_dir", self.results_base_dir)
        )  # base directory where score files will be stored

        self.step_hrs = self.inference_cfg.get("step_hrs", 1)

        # for backward compatibility allow metric_dir to be specified in the run config
        self.results_dir = Path(self.results_base_dir)
        self.runplot_dir = Path(self.runplot_base_dir)
        self.metrics_dir = Path(
            self.eval_cfg.get("metrics_dir", self.metrics_base_dir / "evaluation")
        )

    def get_inference_config(self):
        """
        Load the config associated to the inference run (different from the
        eval_cfg which contains plot and evaluation options.)

        Returns
        -------
        config: dict
            Configuration file from the inference run
        """
        config = {}

        if self.private_paths:
            _logger.info(
                f"Loading config for run {self.run_id} from private paths: {self.private_paths}"
            )
            config = load_merge_configs(self.private_paths, self.run_id, self.mini_epoch)
        else:
            _logger.info(
                f"Loading config for run {self.run_id} from model directory: {self.model_base_dir}"
            )
            config = load_run_config(self.run_id, self.mini_epoch, self.model_base_dir)

        if not isinstance(config, dict | oc.DictConfig):
            _logger.warning("Model config not found. inference config will be empty.")
            config = {}

        return config

    def get_climatology_filename(self, stream: str) -> str | None:
        """
        Get the climatology filename for a given stream from the inference
        configuration.

        Parameters
        ----------
        stream : str
            Name of the data stream.

        Returns
        -------
        path: str | None
            Full climatology path if available, otherwise None.
        """
        stream_dict = self.get_stream(stream)

        clim_data_path = stream_dict.get("climatology_path", None)
        if not clim_data_path:
            clim_base_dir = self.inference_cfg.get("data_path_aux", None)
            clim_fn = next(
                (
                    item.get("climatology_filename")
                    for item in self.inference_cfg.get("streams", [])
                    if item.get("name") == stream
                ),
                None,
            )
            if clim_base_dir and clim_fn:
                clim_data_path = Path(clim_base_dir) / clim_fn
            else:
                _logger.warning(
                    f"No climatology path specified for stream {stream}. Setting climatology to "
                    "NaN. Add 'climatology_path' to evaluation config to use metrics like ACC."
                )

        return str(clim_data_path) if clim_data_path else None

    def get_channels(self, stream: str) -> list[str]:
        """
        Get the list of channels for a given stream from the config.

        Parameters
        ----------
        stream : str
            The name of the stream to get channels for.

        Returns
        -------
        all_channels: list[str]
            A list of channel names.
        """
        _logger.debug(f"Getting channels for stream {stream}...")
        all_channels = self.get_inference_stream_attr(stream, "val_target_channels")
        _logger.debug(f"Channels found in config: {all_channels}")
        return all_channels

    def load_scores(
        self, stream: str, regions: list[str], metrics: dict[str, object]
    ) -> tuple[dict, dict]:
        """
        Load multiple pre-computed scores for a given run, stream and metric
        and epoch.

        Parameters
        ----------
        stream : str
            Stream name.
        regions : list[str]
            Region names.
        metrics : list[str]
            Metric names.

        Returns
        -------
        tuple[dict, dict]
            - local_scores: dictionary of available scores.
            - recomputable_missing_metrics: dictionary of regions and metrics
              that must be recomputed (empty for JSON reader).
        """
        local_scores = {}
        missing_metrics = {}
        for region in regions:
            for metric, parameters in metrics.items():
                score = self.load_single_score(stream, region, metric, parameters)
                if score is None:
                    # all other cases: recompute scores
                    missing_metrics.setdefault(region, {}).update({metric: parameters})
                else:
                    available_data = self.check_availability(stream, score, mode="evaluation")
                    if available_data.score_availability:
                        score = score.sel(
                            sample=available_data.samples,
                            channel=available_data.channels,
                            forecast_step=available_data.fsteps,
                        )
                        local_scores.setdefault(metric, {}).setdefault(region, {}).setdefault(
                            stream, {}
                        )[self.run_id] = score

        recomputable_missing_metrics = self.get_recomputable_metrics(missing_metrics)
        return local_scores, recomputable_missing_metrics

    def load_single_score(
        self, stream: str, region: str, metric: str, parameters: dict | None = None
    ) -> xr.DataArray | None:
        """
        Load a single pre-computed score for a given run, stream and metric.

        Returns
        -------
        score: xr.DataArray or None
            DataArray of the score if found, else None.
        """
        if parameters is None:
            parameters = {}
        score_path = (
            Path(self.metrics_dir)
            / f"{self.run_id}_{stream}_{region}_{metric}_chkpt{self.mini_epoch:05d}.json"
        )
        _logger.debug(f"Looking for: {score_path}")

        score = None
        if score_path.exists():
            with open(score_path) as f:
                data_dict = json.load(f)
                if "scores" not in data_dict:
                    data_dict = {"scores": [data_dict]}
                for score_version in data_dict["scores"]:
                    if score_version["attrs"] == parameters:
                        score = xr.DataArray.from_dict(score_version)
                        break
        return score

    def get_recomputable_metrics(self, metrics: dict) -> dict:
        """
        Determine which metrics can be recomputed.

        Parameters
        ----------
        metrics : dict
            Dictionary mapping regions to missing metrics.

        Returns
        -------
        metrics: dict
            Same as input
        """
        return metrics

    def get_inference_stream_attr(self, stream_name: str, key: str, default=None):
        """
        Get the value of a key for a specific stream from the a model config.

        Parameters:
        ------------
            stream_name: str
                The name of the stream (e.g. 'ERA5').
            key: str
                The key to look up (e.g. 'tokenize_spacetime').
            default: Optional
                Value to return if not found (default: None).

        Returns:
        ------------
            The parameter value if found, otherwise the default.
        """
        for stream in self.inference_cfg.get("streams", []):
            if stream.get("name") == stream_name:
                return stream.get(key, default)
        return default


class WeatherGenJsonReader(WeatherGenReader):
    def __init__(
        self,
        eval_cfg: dict,
        run_id: str,
        private_paths: dict | None = None,
        regions: list[str] | None = None,
        metrics: dict[str, object] | None = None,
    ):
        super().__init__(eval_cfg, run_id, private_paths)
        self.common_coords: dict = self._compute_common_coords(regions, metrics)

    def _compute_common_coords(self, regions: list[str], metrics: list[str]) -> dict:
        # Find common coordinates across streams, regions, metrics.
        streams = list(self.streams)
        coord_names = ["sample", "forecast_step", "ens"]
        all_coords = {name: [] for name in coord_names}
        provenance = {name: defaultdict(list) for name in coord_names}

        for stream in streams:
            for region in regions:
                for metric, parameters in metrics.items():
                    score = self.load_single_score(stream, region, metric, parameters)
                    if score is not None:
                        for name in coord_names:
                            vals = set(score[name].values)
                            all_coords[name].append(vals)
                            for val in vals:
                                provenance[name][val].append((stream, region, metric))

        common_coords = {name: set.intersection(*all_coords[name]) for name in coord_names}

        # Warn about any skipped coordinates
        for name in coord_names:
            skipped = set.union(*all_coords[name]) - common_coords[name]
            if skipped:
                msg_lines = [
                    f"Some {name}(s) were not common across streams, regions, and metrics:"
                ]
                for val in skipped:
                    msg_lines.append(f"  {val} only present in {provenance[name][val]}")
                _logger.warning("\n".join(msg_lines))

        return common_coords

    def get_samples(self) -> set[int]:
        return self.common_coords["sample"]

    def get_forecast_steps(self) -> set[int]:
        return self.common_coords["forecast_step"]

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        return self.common_coords["ens"]

    def get_data(self, *args, **kwargs):
        # TODO this should not be needed, the reader should not even be created if this is the case
        # it can still happen when a particular score was available for a different channel
        assert False, f"Missing JSON data for run {self.run_id}."

    def get_recomputable_metrics(self, metrics):
        _logger.info(
            f"The following metrics have not yet been computed:{metrics}. Use type: zarr for that."
        )
        return {}


class WeatherGenZarrReader(WeatherGenReader):
    def __init__(self, eval_cfg: dict, run_id: str, private_paths: dict | None = None):
        """Data reader class for WeatherGenerator model outputs stored in Zarr format."""
        super().__init__(eval_cfg, run_id, private_paths)

        zarr_ext = self.inference_cfg.get("zarr_store", "zarr")
        # For backwards compatibility, assume zarr store is local (.zarr format).

        fname_zarr = self.results_dir.joinpath(
            f"validation_chkpt{self.mini_epoch:05d}_rank{self.rank:04d}.{zarr_ext}"
        )

        assert fname_zarr.exists(), f"Zarr file {fname_zarr} does not exist."

        assert (zarr_ext == "zarr" and fname_zarr.is_dir()) or (
            zarr_ext == "zip" and fname_zarr.is_file()
        ), (
            f"Zarr file {fname_zarr} has unexpected format. ({zarr_ext}). "
            f"Expected directory for 'zarr' or file for 'zip'."
        )
        self.fname_zarr = fname_zarr

        # Metadata caches — populated lazily on first access
        self._cached_samples: set[int] | None = None
        self._cached_fsteps: set[int] | None = None
        self._cached_streams: set[str] | None = None
        self._cached_ensemble: dict[str, list[str]] = {}
        self._cached_is_gridded: dict[str, bool] = {}

        # I/O threading config
        self._fast_io = eval_cfg.get("fast_io", True)
        self._num_io_threads: int = int(eval_cfg.get("num_io_threads", 8))

        # Fast raw I/O config (direct zarr access, bypasses ZarrIO/dask)
        self._num_io_workers: int = self._resolve_num_io_workers(
            int(eval_cfg.get("num_io_workers", 0))
        )

    @staticmethod
    def _resolve_num_io_workers(requested: int) -> int:
        """Determine safe number of parallel I/O workers.

        Parameters
        ----------
        requested : int
            Value from config (``num_io_workers``).
            0 (the default) means *auto-detect*: use parallel workers only
            when the system has enough headroom.

        On HPC systems the per-user process/thread limit (``ulimit -u``) is
        often shared across all jobs on the node.  Spawning loky workers when
        the limit is almost reached causes "can't start new thread" errors
        that cascade into the fallback path and deadlock it.

        Auto-detection (``requested == 0``):
        * Read ``/proc/self/status`` → current thread count.
        * Read ``ulimit -u`` via ``resource.getrlimit``.
        * If fewer than ``min_headroom`` slots remain → sequential (1).
        * Otherwise cap at ``min(available // 4, cpu_count, 16)``.
        """
        import resource

        if requested > 0:
            return min(requested, os.cpu_count() or 16)

        # Auto-detect safe parallelism
        try:
            # Current threads for this process tree
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("Threads:"):
                        current_threads = int(line.split()[1])
                        break
                else:
                    current_threads = 1

            # System-wide nproc limit for this user
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
            if soft_limit == resource.RLIM_INFINITY:
                soft_limit = 65536

            # Count user processes (rough estimate via /proc)
            import subprocess

            result = subprocess.run(
                ["ps", "-u", str(os.getuid()), "--no-headers", "-o", "pid"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            user_procs = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 0

            available = soft_limit - user_procs
            min_headroom = 64  # keep at least this many slots free

            if available < min_headroom:
                _logger.info(
                    f"Auto-detected low process headroom "
                    f"({available}/{soft_limit} slots free). "
                    f"Using sequential I/O (num_io_workers=1)."
                )
                return 1

            n = min(available // 4, os.cpu_count() or 48, 48)
            n = max(n, 1)
            _logger.info(
                f"Auto-detected process headroom: {available}/{soft_limit} free. "
                f"Using num_io_workers={n}."
            )
            return n

        except Exception as e:
            _logger.debug(f"Could not auto-detect process limits ({e}). Defaulting to sequential.")
            return 1

    def get_data(
        self,
        stream: str,
        samples: list[int] | None = None,
        fsteps: list[str] | None = None,
        channels: list[str] | None = None,
        ensemble: list[str] | None = None,
    ) -> ReaderOutput:
        """
        Retrieve prediction and target data for a given run from the Zarr store.

        Parameters
        ----------
        cfg :
            Configuration dictionary containing all information for the evaluation.
        results_dir : Path
            Directory where the inference results are stored.
            Expected scheme `<results_base_dir>/<run_id>`.
        stream :
            Stream name to retrieve data for.
        samples :
            List of sample indices to retrieve. If None, all samples are retrieved.
        fsteps :
            List of forecast steps to retrieve. If None, all forecast steps are retrieved.
        channels :
            List of channel names to retrieve. If None, all channels are retrieved.

        Returns
        -------
        out: ReaderOutput
            A dataclass containing:
            - target: Dictionary of xarray DataArrays for targets, indexed by forecast step.
            - prediction: Dictionary of xarray DataArrays for predictions, indexed by forecast step.
        """
        stream_cfg = self.get_stream(stream)
        all_channels = self.get_channels(stream)
        _logger.info(f"RUN {self.run_id}: Processing stream {stream}...")

        fsteps = self.get_forecast_steps() if fsteps is None else fsteps

        # TODO: Avoid conversion of fsteps and sample to integers (as obtained from the ZarrIO)
        fsteps = sorted([int(fstep) for fstep in fsteps])
        samples = samples or sorted([int(sample) for sample in self.get_samples()])
        channels = channels or stream_cfg.get("channels", all_channels)
        channels = to_list(channels)

        ensemble = ensemble or self.get_ensemble(stream)
        ensemble = to_list(ensemble)

        is_gridded_data = self.is_gridded_data(stream)

        with zarrio_reader(self.fname_zarr) as zio:
            # ----------------------------------------------------------
            # Phase 1: Dispatch ALL (sample, fstep) reads at once
            # ----------------------------------------------------------
            n_total = len(samples) * len(fsteps)
            effective_threads = min(self._num_io_threads, n_total)

            # ZipStore detection: Python's zipfile.ZipFile is NOT thread-safe
            # when shared, but each thread can safely open its own handle.
            is_zip_store = hasattr(zio, "_store") and "zip" in type(zio._store).__name__.lower()

            # # For ZipStore, never use parallel workers in get_data().
            # # ProcessPoolExecutor requires spawning new processes, which
            # # routinely fails on HPC nodes with tight ulimit -u (shared
            # # across all user jobs).  The spawned workers also need to
            # # pickle/unpickle full xarray DataArrays over IPC, making them
            # # slower than sequential reads in practice.
            # if is_zip_store:
            #     effective_threads = 1

            _logger.info(
                f"RUN {self.run_id} - {stream}: Loading {len(samples)} samples × "
                f"{len(fsteps)} fsteps = {n_total} items "
                f"(workers={effective_threads}, zip={is_zip_store})..."
            )

            # results_by_fstep[fstep] = list of (sample, target, pred, valid_times)
            results_by_fstep: dict[int, list] = {f: [] for f in fsteps}

            if effective_threads > 1:
                # Choose executor + loader based on store type:
                #
                # LocalStore → ThreadPoolExecutor + shared zio
                #   zarr LocalStore reads release the GIL (C-level I/O),
                #   so threads give real parallelism. One shared zio is safe.
                #
                # ZipStore → ProcessPoolExecutor + per-process zio
                #   zarr ZipStore reads go through Python's zipfile module
                #   which does NOT release the GIL, plus each ZipStore
                #   instance holds a threading.RLock that serialises reads.
                #   Threads are useless here — only separate processes
                #   (each with their own GIL) give real parallelism.
                if is_zip_store:
                    pool_cls = ProcessPoolExecutor
                    pool_kwargs = {
                        "max_workers": effective_threads,
                        "initializer": _init_worker_zio,
                        "initargs": (self.fname_zarr,),
                    }
                    submit_args = lambda s, f: (
                        _load_single_sample_own_context,
                        self.fname_zarr,
                        s,
                        stream,
                        f,
                        ensemble,
                        is_gridded_data,
                    )
                else:
                    pool_cls = ThreadPoolExecutor
                    pool_kwargs = {"max_workers": effective_threads}
                    submit_args = lambda s, f: (
                        _load_single_sample,
                        zio,
                        s,
                        stream,
                        f,
                        ensemble,
                        is_gridded_data,
                    )

                try:
                    with pool_cls(**pool_kwargs) as executor:
                        futures = {
                            executor.submit(*submit_args(s, f)): (f, s)
                            for f in fsteps
                            for s in samples
                        }
                        for future in tqdm(
                            as_completed(futures),
                            total=n_total,
                            desc=f"Loading {self.run_id} - {stream}",
                        ):
                            result = future.result()
                            if result is not None:
                                fstep_r, sample_r, target, pred, vt = result
                                results_by_fstep[fstep_r].append((sample_r, target, pred, vt))
                except (RuntimeError, OSError) as pool_err:
                    _logger.warning(
                        f"Parallel pool failed ({pool_err}). Falling back to sequential loading."
                    )
                    effective_threads = 0  # force sequential below

            if effective_threads <= 1:
                for f in fsteps:
                    for s in tqdm(samples, desc=f"Loading {self.run_id} - {stream} - fstep {f}"):
                        result = _load_single_sample(
                            zio,
                            s,
                            stream,
                            f,
                            ensemble,
                            is_gridded_data,
                        )
                        if result is None:
                            continue
                        _, sample_r, target, pred, vt = result
                        results_by_fstep[f].append((sample_r, target, pred, vt))

        # ----------------------------------------------------------
        # Phase 2: Reassemble per-fstep (outside the zarr context)
        # ----------------------------------------------------------
        da_tars, da_preds = [], []
        fsteps_final = []

        for fstep in fsteps:
            per_fstep = results_by_fstep[fstep]
            if not per_fstep:
                _logger.info(f"[{self.run_id} - {stream}] No valid data for fstep {fstep}.")
                continue

            # Sort by sample index for deterministic output
            per_fstep.sort(key=lambda x: x[0])

            da_tars_fs = [r[1] for r in per_fstep]
            da_preds_fs = [r[2] for r in per_fstep]
            valid_times_fs = [r[3] for r in per_fstep if r[3] is not None]

            fsteps_final.append(valid_times_fs if valid_times_fs else fstep)

            _logger.debug(
                f"Concatenating targets and predictions for stream {stream}, "
                f"forecast_step {fstep}..."
            )

            if is_gridded_data:
                da_preds_fs = _split_by_valid_time(da_preds_fs)
                da_tars_fs = _split_by_valid_time(da_tars_fs)
            else:
                da_tars_fs = xr.concat(
                    da_tars_fs, dim="ipoint", coords="different", compat="equals"
                )
                da_preds_fs = xr.concat(
                    da_preds_fs, dim="ipoint", coords="different", compat="equals"
                )

            da_tars.append(da_tars_fs)
            da_preds.append(da_preds_fs)

        # ----------------------------------------------------------
        # Phase 3: Channel selection and coordinate assignment
        # ----------------------------------------------------------
        da_tars_dict, da_preds_dict = {}, {}
        i = 1

        for fstep, da_t, da_p in zip(fsteps_final, da_tars, da_preds, strict=True):
            with_substeps = isinstance(da_t, list)
            items = zip(da_t, da_p, strict=True) if with_substeps else [(da_t, da_p)]

            for t, p in items:
                t, p = _select_channels(t, p, stream, channels, stream_cfg)

                if is_gridded_data:
                    t = _add_lead_time_coord(t)
                    p = _add_lead_time_coord(p)

                    p = _scale_z_channels(p, stream)
                    t = _scale_z_channels(t, stream)

                if with_substeps:
                    t = t.assign_coords(forecast_step=i)
                    p = p.assign_coords(forecast_step=i)
                    da_tars_dict[i] = t
                    da_preds_dict[i] = p
                    i += 1
                else:
                    da_tars_dict[int(fstep)] = t
                    da_preds_dict[int(fstep)] = p

        return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)

    def get_data_raw(
        self,
        stream: str,
        samples: list[int] | None = None,
        fsteps: list[int] | None = None,
        channels: list[str] | None = None,
        ensemble: list[str] | None = None,
    ) -> ReaderOutput:
        """
        Fast-path data loading using direct zarr array access and joblib parallelism.

        Bypasses the ZarrIO → OutputDataset → as_xarray → dask pipeline for ~20× speedup
        per (sample, fstep) read.  Produces identical ReaderOutput to get_data().

        Falls back to get_data() on any error (unexpected zarr layout, missing
        attributes, shape mismatch, etc.).

        Parameters
        ----------
        stream : str
            Stream name to retrieve data for.
        samples : list[int] | None
            Sample indices. If None, all samples are retrieved.
        fsteps : list[int] | None
            Forecast steps. If None, all forecast steps are retrieved.
        channels : list[str] | None
            Channel names to retrieve. If None, all channels are retrieved.
        ensemble : list[str] | None
            Ensemble members to select, or ["mean"] to average. If None, all.

        Returns
        -------
        ReaderOutput
            Identical structure to get_data() output.
        """
        # Choose implementation based on store type.
        # ZipStore benefits from dispatching all samples at once (each
        # reading all fsteps in a single _read_sample_raw call) because
        # opening a ZipStore requires parsing the central-directory – a
        # cost that is amortised over all fsteps inside a single call.
        is_zip = str(self.fname_zarr).endswith(".zip")

        try:
            if is_zip:
                return self._get_data_raw_zip_impl(
                    stream,
                    samples,
                    fsteps,
                    channels,
                    ensemble,
                )
            return self._get_data_raw_impl(stream, samples, fsteps, channels, ensemble)
        except Exception as e:
            _logger.warning(
                f"Fast I/O failed for {self.run_id} - {stream}: {e}. "
                f"Falling back to standard get_data()."
            )
            # Ensure the loky reusable executor is fully shut down before
            # the fallback tries to create its own process/thread pools.
            # Without this, stale loky workers can exhaust the OS
            # process/thread limit and deadlock the fallback path.
            try:
                get_reusable_executor().shutdown(wait=True)
            except Exception:
                pass
            return self.get_data(stream, samples, fsteps, channels, ensemble)

    def _get_data_raw_impl(
        self,
        stream: str,
        samples: list[int] | None,
        fsteps: list[int] | None,
        channels: list[str] | None,
        ensemble: list[str] | None,
    ) -> ReaderOutput:
        """Internal implementation of get_data_raw.

        Processes one forecast step at a time to keep peak memory bounded at
        ``n_samples × 1 × n_ipoints × n_channels × 4 bytes`` rather than
        ``n_samples × n_fsteps × …``.
        """
        stream_cfg = self.get_stream(stream)
        all_channels = self.get_channels(stream)
        is_gridded_data = self.is_gridded_data(stream)
        _logger.info(f"RUN {self.run_id}: Processing stream {stream} (fast raw I/O)...")

        fsteps = sorted(int(f) for f in (fsteps or self.get_forecast_steps()))
        samples = sorted(int(s) for s in (samples or self.get_samples()))
        channels = channels or stream_cfg.get("channels", all_channels)
        channels = to_list(channels)
        ensemble = ensemble or self.get_ensemble(stream)
        ensemble = to_list(ensemble)

        # ---- Pre-compute channel indices into the zarr data arrays ----
        channel_idxs = list(range(len(all_channels)))
        read_channels = all_channels

        # ---- Read coordinates and metadata once ----
        zarr_path = str(self.fname_zarr)
        is_zip = zarr_path.endswith(".zip")
        coords_raw, zarr_channels, _ = _read_coords_and_meta(zarr_path, stream, fsteps[0], is_zip)
        if zarr_channels:
            read_channels = zarr_channels
            channel_idxs = None

        # ---- Early channel selection (skip unrequested channels) ----
        channel_idxs, read_channels = _compute_early_channel_selection(
            read_channels, channels, stream_cfg
        )

        lat = coords_raw[:, 0].astype(np.float64)
        lon = coords_raw[:, 1].astype(np.float64)

        need_per_sample_coords = not is_gridded_data

        n_workers = min(self._num_io_workers, len(samples))

        # Always use "loky" backend.  The loky reusable-executor is a
        # **persistent process pool** that survives across Parallel() calls,
        # so iterating over fsteps does not leak threads/processes.
        # The previous "threading" backend created a *new* ThreadPoolExecutor
        # per fstep, which quickly exhausted the OS thread limit ("can't
        # start new thread") on systems with tight ulimits.
        backend = "loky"

        _logger.info(
            f"RUN {self.run_id} - {stream}: Loading {len(samples)} samples × "
            f"{len(fsteps)} fsteps via raw zarr I/O "
            f"(workers={n_workers}, backend={backend})..."
        )

        # Pre-fetch ensemble names once (outside the per-fstep loop)
        all_ens: list[str] | None = None
        if ensemble != ["mean"]:
            all_ens = self.get_ensemble(stream)

        # ---- Process one forecast step at a time to limit peak memory ----
        # Each iteration: read all samples for one fstep → build DataArrays
        # → store → free raw arrays before the next fstep.
        da_tars_dict, da_preds_dict = {}, {}
        fstep_counter = 1  # for sub-step numbering
        source_interval_starts: np.ndarray | None = None

        for fi, fs in enumerate(fsteps):
            _logger.info(
                f"RUN {self.run_id} - {stream}: Reading fstep {fs} ({fi + 1}/{len(fsteps)})..."
            )

            # Dispatch parallel reads for this single fstep.
            # The loky reusable executor pool persists across iterations,
            # so no new processes/threads are spawned per fstep.
            if n_workers > 1:
                try:
                    results = Parallel(n_jobs=n_workers, backend=backend, verbose = 5 )(
                        delayed(_read_sample_raw)(
                            zarr_path,
                            s,
                            stream,
                            [fs],
                            channel_idxs,
                            is_zip,
                            read_coords=need_per_sample_coords,
                            is_gridded=is_gridded_data,
                        )
                        for s in samples
                    )
                except (RuntimeError, OSError) as pool_err:
                    _logger.warning(
                        f"Parallel pool failed on fstep {fs} ({pool_err}). "
                        f"Switching to sequential reads for remaining fsteps."
                    )
                    try:
                        get_reusable_executor().shutdown(wait=True)
                    except Exception:
                        pass
                    n_workers = 1  # sequential for this and all subsequent fsteps
                    results = [
                        _read_sample_raw(
                            zarr_path,
                            s,
                            stream,
                            [fs],
                            channel_idxs,
                            is_zip,
                            read_coords=need_per_sample_coords,
                            is_gridded=is_gridded_data,
                        )
                        for s in samples
                    ]
            else:
                results = [
                    _read_sample_raw(
                        zarr_path,
                        s,
                        stream,
                        [fs],
                        channel_idxs,
                        is_zip,
                        read_coords=need_per_sample_coords,
                        is_gridded=is_gridded_data,
                    )
                    for s in samples
                ]

            # results[i] = (preds_all, targets_all, times_all, meta)
            # With a single fstep, preds_all / targets_all have 1+ entries
            # (>1 only if there are sub-steps).

            # ---- Extract source_interval once (from first fstep) ----
            if source_interval_starts is None:
                si_list = []
                for i in range(len(samples)):
                    meta = results[i][3]
                    si = meta.get("source_interval", {})
                    start_str = si.get("start", None)
                    if start_str is not None:
                        si_list.append(np.datetime64(start_str, "ns"))
                    else:
                        si_list.append(np.datetime64("NaT", "ns"))
                source_interval_starts = np.array(si_list)

            # ---- Determine sub-steps for this fstep ----
            n_substeps = results[0][3]["n_substeps"]  # list with 1 entry
            n_sub = n_substeps[0]

            for sub_idx in range(n_sub):
                list_idx = sub_idx  # flat index into the single-fstep results

                tars_list = [results[i][1][list_idx] for i in range(len(samples))]
                preds_list = [results[i][0][list_idx] for i in range(len(samples))]

                # Per-sample valid_times
                per_sample_valid_times = []
                for i in range(len(samples)):
                    time_entry = results[i][2][0]  # index 0: this is the only fstep
                    if n_sub > 1 and sub_idx < len(time_entry):
                        per_sample_valid_times.append(np.datetime64(time_entry[sub_idx], "ns"))
                    elif len(time_entry) > 0:
                        per_sample_valid_times.append(np.datetime64(time_entry[0], "ns"))
                    else:
                        per_sample_valid_times.append(np.datetime64("NaT", "ns"))

                if is_gridded_data:
                    da_tar, da_pred = self._build_gridded_dataarrays(
                        tars_list,
                        preds_list,
                        samples,
                        read_channels,
                        lat,
                        lon,
                        per_sample_valid_times,
                        source_interval_starts,
                        fs if n_sub == 1 else fstep_counter,
                        ensemble,
                        all_ens,
                    )
                else:
                    per_sample_coords = [
                        results[i][3].get("coords", None) for i in range(len(samples))
                    ]
                    # For scatter data, times_all[0] is the full
                    # per-observation times array (not unique-only).
                    per_sample_obs_times = [
                        results[i][2][0]  # fstep index 0 (single fstep)
                        for i in range(len(samples))
                    ]
                    da_tar, da_pred = self._build_scatter_dataarrays(
                        tars_list,
                        preds_list,
                        samples,
                        read_channels,
                        per_sample_valid_times,
                        source_interval_starts,
                        fs if n_sub == 1 else fstep_counter,
                        ensemble,
                        all_ens,
                        per_sample_coords,
                        coords_raw,
                        per_sample_obs_times=per_sample_obs_times,
                    )

                # Free raw numpy lists before post-processing creates more copies
                del tars_list, preds_list

                da_tar, da_pred = _select_channels(da_tar, da_pred, stream, channels, stream_cfg)

                if is_gridded_data:
                    da_tar = _add_lead_time_coord(da_tar)
                    da_pred = _add_lead_time_coord(da_pred)
                    da_pred = _scale_z_channels(da_pred, stream)
                    da_tar = _scale_z_channels(da_tar, stream)

                if n_sub > 1:
                    da_tar = da_tar.assign_coords(forecast_step=fstep_counter)
                    da_pred = da_pred.assign_coords(forecast_step=fstep_counter)
                    da_tars_dict[fstep_counter] = da_tar
                    da_preds_dict[fstep_counter] = da_pred
                    fstep_counter += 1
                else:
                    da_tars_dict[int(fs)] = da_tar
                    da_preds_dict[int(fs)] = da_pred

            # Free raw results for this fstep before reading the next
            del results

        # Shut down the loky reusable worker pool once after all fsteps
        # to release semaphores and temp folders.
        if n_workers > 1:
            get_reusable_executor().shutdown(wait=True)

        _logger.info(
            f"RUN {self.run_id} - {stream}: Raw I/O complete. "
            f"{len(da_tars_dict)} forecast entries loaded."
        )
        return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)

    def _get_data_raw_zip_impl(
        self,
        stream: str,
        samples: list[int] | None,
        fsteps: list[int] | None,
        channels: list[str] | None,
        ensemble: list[str] | None,
    ) -> ReaderOutput:
        """ZipStore-optimised implementation of get_data_raw.

        Unlike ``_get_data_raw_impl`` which processes one fstep at a time
        (serial fstep loop, parallel samples), this method dispatches **all**
        ``(sample, fstep)`` pairs in a single ``joblib.Parallel`` call.

        Why this is faster for ZipStore
        --------------------------------
        * Each ``_read_sample_raw`` call opens its own ``zarr.storage.ZipStore``
          handle (required because ``zipfile.ZipFile`` is not thread-safe).
          The cost of parsing the zip central-directory is amortised over all
          fsteps requested for that sample in a single call.
        * A single ``Parallel`` dispatch avoids repeated loky pool
          synchronisation barriers (one per fstep in the old approach).
        * Memory stays bounded: each worker returns numpy arrays for one
          sample (all fsteps), which are small compared to the full dataset.

        Peak memory is ``n_workers × n_fsteps × n_ipoints × n_channels × 4``
        bytes (one worker's payload in flight at a time), versus the old
        approach which was ``n_samples × 1 × …`` per fstep iteration.
        For typical evaluation sizes this is comparable.
        """
        stream_cfg = self.get_stream(stream)
        all_channels = self.get_channels(stream)
        is_gridded_data = self.is_gridded_data(stream)
        _logger.info(
            f"RUN {self.run_id}: Processing stream {stream} (fast raw I/O – ZipStore parallel)..."
        )

        fsteps = sorted(int(f) for f in (fsteps or self.get_forecast_steps()))
        samples = sorted(int(s) for s in (samples or self.get_samples()))
        channels = channels or stream_cfg.get("channels", all_channels)
        channels = to_list(channels)
        ensemble = ensemble or self.get_ensemble(stream)
        ensemble = to_list(ensemble)

        # ---- Pre-compute channel indices into the zarr data arrays ----
        channel_idxs = list(range(len(all_channels)))
        read_channels = all_channels

        # ---- Read coordinates and metadata once ----
        zarr_path = str(self.fname_zarr)
        is_zip = zarr_path.endswith(".zip")
        coords_raw, zarr_channels, _ = _read_coords_and_meta(zarr_path, stream, fsteps[0], is_zip)
        if zarr_channels:
            read_channels = zarr_channels
            channel_idxs = None

        # ---- Early channel selection (skip unrequested channels) ----
        channel_idxs, read_channels = _compute_early_channel_selection(
            read_channels, channels, stream_cfg
        )

        lat = coords_raw[:, 0].astype(np.float64)
        lon = coords_raw[:, 1].astype(np.float64)

        need_per_sample_coords = not is_gridded_data

        n_workers = min(self._num_io_workers, len(samples))
        backend = "loky"

        # Pre-fetch ensemble names once
        all_ens: list[str] | None = None
        if ensemble != ["mean"]:
            all_ens = self.get_ensemble(stream)

        _logger.info(
            f"RUN {self.run_id} - {stream}: Loading {len(samples)} samples × "
            f"{len(fsteps)} fsteps via ZipStore-parallel raw zarr I/O "
            f"(workers={n_workers}, backend={backend})..."
        )

        # ------------------------------------------------------------------
        # Single Parallel dispatch: each call reads ALL fsteps for one sample.
        # This means each loky worker opens the ZipStore once and reads all
        # fsteps sequentially inside that single handle – amortising the
        # central-directory parse cost.
        # ------------------------------------------------------------------
        if n_workers > 1:
            try:
                results = Parallel(n_jobs=n_workers, backend=backend, verbose = 5 )(
                    delayed(_read_sample_raw)(
                        zarr_path,
                        s,
                        stream,
                        fsteps,
                        channel_idxs,
                        is_zip,
                        read_coords=need_per_sample_coords,
                        is_gridded=is_gridded_data,
                    )
                    for s in samples
                )
            except (RuntimeError, OSError) as pool_err:
                _logger.warning(
                    f"ZipStore parallel pool failed ({pool_err}). Switching to sequential reads."
                )
                try:
                    get_reusable_executor().shutdown(wait=True)
                except Exception:
                    pass
                n_workers = 1
                results = [
                    _read_sample_raw(
                        zarr_path,
                        s,
                        stream,
                        fsteps,
                        channel_idxs,
                        is_zip,
                        read_coords=need_per_sample_coords,
                        is_gridded=is_gridded_data,
                    )
                    for s in samples
                ]
        else:
            results = [
                _read_sample_raw(
                    zarr_path,
                    s,
                    stream,
                    fsteps,
                    channel_idxs,
                    is_zip,
                    read_coords=need_per_sample_coords,
                    is_gridded=is_gridded_data,
                )
                for s in samples
            ]

        # results[i] = (preds_all, targets_all, times_all, meta)
        # where preds_all/targets_all have entries for each fstep (and
        # potentially multiple sub-steps per fstep).

        # ---- Extract source_interval from the first sample ----
        si_list = []
        for i in range(len(samples)):
            meta = results[i][3]
            si = meta.get("source_interval", {})
            start_str = si.get("start", None)
            if start_str is not None:
                si_list.append(np.datetime64(start_str, "ns"))
            else:
                si_list.append(np.datetime64("NaT", "ns"))
        source_interval_starts = np.array(si_list)

        # ---- Reassemble per-fstep ----
        # Each result contains arrays for ALL fsteps.  We need to slice
        # into them.  The n_substeps list tells us how many sub-step
        # arrays each fstep produced.
        n_substeps_per_fstep = results[0][3]["n_substeps"]  # list of len(fsteps)

        da_tars_dict, da_preds_dict = {}, {}
        fstep_counter = 1

        # Build a flat-index offset for each fstep into the results lists
        offsets = []
        off = 0
        for ns in n_substeps_per_fstep:
            offsets.append(off)
            off += ns

        for fi, fs in enumerate(fsteps):
            n_sub = n_substeps_per_fstep[fi]
            base_off = offsets[fi]

            for sub_idx in range(n_sub):
                list_idx = base_off + sub_idx

                tars_list = [results[i][1][list_idx] for i in range(len(samples))]
                preds_list = [results[i][0][list_idx] for i in range(len(samples))]

                # Per-sample valid_times
                per_sample_valid_times = []
                for i in range(len(samples)):
                    time_entry = results[i][2][fi]  # fi-th fstep
                    if n_sub > 1 and sub_idx < len(time_entry):
                        per_sample_valid_times.append(np.datetime64(time_entry[sub_idx], "ns"))
                    elif len(time_entry) > 0:
                        per_sample_valid_times.append(np.datetime64(time_entry[0], "ns"))
                    else:
                        per_sample_valid_times.append(np.datetime64("NaT", "ns"))

                if is_gridded_data:
                    da_tar, da_pred = self._build_gridded_dataarrays(
                        tars_list,
                        preds_list,
                        samples,
                        read_channels,
                        lat,
                        lon,
                        per_sample_valid_times,
                        source_interval_starts,
                        fs if n_sub == 1 else fstep_counter,
                        ensemble,
                        all_ens,
                    )
                else:
                    per_sample_coords = [
                        results[i][3].get("coords", None) for i in range(len(samples))
                    ]
                    # For scatter data, times_all[fi] is the full
                    # per-observation times array (not unique-only).
                    per_sample_obs_times = [
                        results[i][2][fi]  # fi-th fstep
                        for i in range(len(samples))
                    ]
                    da_tar, da_pred = self._build_scatter_dataarrays(
                        tars_list,
                        preds_list,
                        samples,
                        read_channels,
                        per_sample_valid_times,
                        source_interval_starts,
                        fs if n_sub == 1 else fstep_counter,
                        ensemble,
                        all_ens,
                        per_sample_coords,
                        coords_raw,
                        per_sample_obs_times=per_sample_obs_times,
                    )

                del tars_list, preds_list

                da_tar, da_pred = _select_channels(da_tar, da_pred, stream, channels, stream_cfg)

                if is_gridded_data:
                    da_tar = _add_lead_time_coord(da_tar)
                    da_pred = _add_lead_time_coord(da_pred)
                    da_pred = _scale_z_channels(da_pred, stream)
                    da_tar = _scale_z_channels(da_tar, stream)

                if n_sub > 1:
                    da_tar = da_tar.assign_coords(forecast_step=fstep_counter)
                    da_pred = da_pred.assign_coords(forecast_step=fstep_counter)
                    da_tars_dict[fstep_counter] = da_tar
                    da_preds_dict[fstep_counter] = da_pred
                    fstep_counter += 1
                else:
                    da_tars_dict[int(fs)] = da_tar
                    da_preds_dict[int(fs)] = da_pred

        # Free raw results
        del results

        # Shut down loky pool
        if n_workers > 1:
            get_reusable_executor().shutdown(wait=True)

        _logger.info(
            f"RUN {self.run_id} - {stream}: ZipStore-parallel raw I/O complete. "
            f"{len(da_tars_dict)} forecast entries loaded."
        )
        return ReaderOutput(target=da_tars_dict, prediction=da_preds_dict)

    ######## DataArray construction helpers for get_data_raw ########

    @staticmethod
    def _build_gridded_dataarrays(
        tars_list: list[np.ndarray],
        preds_list: list[np.ndarray],
        samples: list[int],
        read_channels: list[str],
        lat: np.ndarray,
        lon: np.ndarray,
        per_sample_valid_times: list[np.datetime64],
        source_interval_starts: np.ndarray,
        forecast_step_val: int,
        ensemble: list[str],
        all_ens: list[str] | None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Build DataArrays for gridded data by stacking samples along a new axis.

        All samples share the same grid, so np.stack works directly.

        Parameters
        ----------
        per_sample_valid_times : list[np.datetime64]
            One valid_time per sample.  Each sample represents a different
            forecast initialisation, so valid_time differs across samples
            even for the same forecast step.

        Returns
        -------
        da_tar, da_pred : xr.DataArray
        """
        n_samples = len(samples)
        n_ipoints = tars_list[0].shape[0]
        sub_lat = lat[:n_ipoints]
        sub_lon = lon[:n_ipoints]

        tars_stacked = np.stack(tars_list, axis=0)  # (n_samples, n_ipoints, n_channels)
        preds_stacked = np.stack(preds_list, axis=0)  # (n_samples, n_ipoints, n_channels[, n_ens])

        # valid_time must be 2D (sample, ipoint) to match the shape produced by
        # get_data() → _force_consistent_grids → xr.concat(dim="sample").
        # _add_lead_time_coord computes lead_time = valid_time - source_interval_start
        # and needs both arrays to broadcast as (sample, ipoint).
        # Each sample has its OWN valid_time (different initialisation dates),
        # so we build a 2D array where row i is filled with sample i's time.
        vt_col = np.array(per_sample_valid_times, dtype="datetime64[ns]")  # (n_samples,)
        valid_time_2d = np.broadcast_to(
            vt_col[:, np.newaxis],  # (n_samples, 1)
            (n_samples, n_ipoints),
        ).copy()  # copy: broadcast arrays are read-only

        base_coords = {
            "sample": samples,
            "ipoint": np.arange(n_ipoints),
            "channel": read_channels,
            "lat": ("ipoint", sub_lat),
            "lon": ("ipoint", sub_lon),
            "valid_time": (("sample", "ipoint"), valid_time_2d),
            "source_interval_start": ("sample", source_interval_starts.copy()),
            "forecast_step": forecast_step_val,
        }

        da_tar = xr.DataArray(
            tars_stacked,
            dims=["sample", "ipoint", "channel"],
            coords=base_coords,
        )

        da_pred = WeatherGenZarrReader._build_pred_dataarray(
            preds_stacked,
            base_coords,
            ensemble,
            all_ens,
        )

        return da_tar, da_pred

    @staticmethod
    def _build_scatter_dataarrays(
        tars_list: list[np.ndarray],
        preds_list: list[np.ndarray],
        samples: list[int],
        read_channels: list[str],
        per_sample_valid_times: list[np.datetime64],
        source_interval_starts: np.ndarray,
        forecast_step_val: int,
        ensemble: list[str],
        all_ens: list[str] | None,
        per_sample_coords: list[np.ndarray | None],
        coords_fallback: np.ndarray,
        per_sample_obs_times: list[np.ndarray] | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Build DataArrays for non-gridded (scatter) data.

        Samples may have different ipoint counts, so we concatenate along
        the ipoint dimension — matching the get_data() behavior for scatter data.

        Parameters
        ----------
        per_sample_valid_times : list[np.datetime64]
            One representative valid_time per sample (used as fallback when
            per-observation times are not available).
        per_sample_coords : list[np.ndarray | None]
            Per-sample coordinate arrays read from zarr (shape (n_ip, 2) each).
            Falls back to coords_fallback when None.
        coords_fallback : np.ndarray
            Reference coords from sample 0, used as fallback.
        per_sample_obs_times : list[np.ndarray] | None
            Per-sample arrays of observation times, shape (n_ip,) each.
            When provided, each observation gets its actual timestamp;
            otherwise the single per_sample_valid_times value is broadcast.

        Returns
        -------
        da_tar, da_pred : xr.DataArray
        """
        per_sample_tars = []
        per_sample_preds = []

        for si, sample in enumerate(samples):
            n_ip = tars_list[si].shape[0]
            tar_data = tars_list[si]  # (n_ip, n_channels)
            pred_data = preds_list[si]  # (n_ip, n_channels[, n_ens])

            # Use per-sample coords if available, otherwise fall back to reference
            sc = per_sample_coords[si] if si < len(per_sample_coords) else None
            if sc is not None and len(sc) >= n_ip:
                sample_lat = sc[:n_ip, 0].astype(np.float64)
                sample_lon = sc[:n_ip, 1].astype(np.float64)
            elif coords_fallback is not None and n_ip <= len(coords_fallback):
                sample_lat = coords_fallback[:n_ip, 0].astype(np.float64)
                sample_lon = coords_fallback[:n_ip, 1].astype(np.float64)
            else:
                sample_lat = np.full(n_ip, np.nan)
                sample_lon = np.full(n_ip, np.nan)

            vt_arr = (
                per_sample_obs_times[si][:n_ip].astype("datetime64[ns]")
                if per_sample_obs_times is not None and si < len(per_sample_obs_times)
                else np.full(n_ip, per_sample_valid_times[si], dtype="datetime64[ns]")
            )
            si_start = source_interval_starts[si]

            sample_coords = {
                "ipoint": np.arange(n_ip),
                "channel": read_channels,
                "lat": ("ipoint", sample_lat),
                "lon": ("ipoint", sample_lon),
                "valid_time": ("ipoint", vt_arr),
                "source_interval_start": si_start,
                "forecast_step": forecast_step_val,
                "sample": sample,
            }

            da_t = xr.DataArray(
                tar_data,
                dims=["ipoint", "channel"],
                coords=sample_coords,
            )
            per_sample_tars.append(da_t)

            # Handle ensemble for predictions
            if pred_data.ndim == 3:
                if ensemble == ["mean"]:
                    pred_data = pred_data.mean(axis=-1)
                    pred_coords = dict(sample_coords)
                    da_p = xr.DataArray(
                        pred_data,
                        dims=["ipoint", "channel"],
                        coords=pred_coords,
                    )
                else:
                    ens_idxs = (
                        [all_ens.index(e) for e in ensemble]
                        if all_ens
                        else list(range(pred_data.shape[-1]))
                    )
                    pred_data = pred_data[:, :, ens_idxs]
                    pred_coords = dict(sample_coords)
                    pred_coords["ens"] = ensemble
                    da_p = xr.DataArray(
                        pred_data,
                        dims=["ipoint", "channel", "ens"],
                        coords=pred_coords,
                    )
            else:
                da_p = xr.DataArray(
                    pred_data,
                    dims=["ipoint", "channel"],
                    coords=sample_coords,
                )
            per_sample_preds.append(da_p)

        # Concatenate along ipoint (like get_data() does for non-gridded)
        da_tar = xr.concat(per_sample_tars, dim="ipoint", coords="different", compat="equals")
        da_pred = xr.concat(per_sample_preds, dim="ipoint", coords="different", compat="equals")

        return da_tar, da_pred

    @staticmethod
    def _build_pred_dataarray(
        preds_stacked: np.ndarray,
        base_coords: dict,
        ensemble: list[str],
        all_ens: list[str] | None,
    ) -> xr.DataArray:
        """Build prediction DataArray, handling ensemble dimension.

        Parameters
        ----------
        preds_stacked : np.ndarray
            Shape (n_samples, n_ipoints, n_channels[, n_ens]).
        base_coords : dict
            Coordinate dict (without ens).
        ensemble : list[str]
            Requested ensemble members or ["mean"].
        all_ens : list[str] | None
            All ensemble member names from zarr (needed for index mapping).
        """
        if preds_stacked.ndim == 4:
            if ensemble == ["mean"]:
                # Average over ensemble axis, drop ens coordinate
                # (matches get_data() which does .mean("ens").squeeze())
                preds_stacked = preds_stacked.mean(axis=-1)
                return xr.DataArray(
                    preds_stacked,
                    dims=["sample", "ipoint", "channel"],
                    coords=base_coords,
                )
            else:
                # Select requested ensemble members by index
                if all_ens is not None:
                    ens_idxs = [all_ens.index(e) for e in ensemble]
                    preds_stacked = preds_stacked[:, :, :, ens_idxs]
                pred_coords = dict(base_coords)
                pred_coords["ens"] = ensemble
                return xr.DataArray(
                    preds_stacked,
                    dims=["sample", "ipoint", "channel", "ens"],
                    coords=pred_coords,
                )
        else:
            # No ensemble dim
            return xr.DataArray(
                preds_stacked,
                dims=["sample", "ipoint", "channel"],
                coords=base_coords,
            )

    ######## reader utils ########

    def get_stream(self, stream: str):
        """
        returns the dictionary associated to a particular stream.
        Returns an empty dictionary if the stream does not exist in the Zarr file.

        Parameters
        ----------
        stream:
            the stream name

        Returns
        -------
            The config dictionary associated to that stream
        """
        if self._cached_streams is None:
            with zarrio_reader(self.fname_zarr) as zio:
                self._cached_streams = set(zio.streams)

        if stream in self._cached_streams:
            return self.eval_cfg.streams.get(stream, {})
        return {}

    def get_samples(self) -> set[int]:
        """Get the set of sample indices from the Zarr file."""
        if self._cached_samples is None:
            with zarrio_reader(self.fname_zarr) as zio:
                self._cached_samples = set(int(s) for s in zio.samples)
        return self._cached_samples

    def get_forecast_steps(self) -> set[int]:
        """Get the set of forecast steps from the Zarr file."""
        if self._cached_fsteps is None:
            with zarrio_reader(self.fname_zarr) as zio:
                self._cached_fsteps = set(int(f) for f in zio.forecast_steps)
        return self._cached_fsteps

    def get_forecast_substep_valid_times(self, stream: str) -> set[str]:
        """Get the set of forecast times from the Zarr file."""
        if not self.is_gridded_data(stream):
            _logger.warning(f"Stream {stream} is not gridded. Forecast times cannot be retrieved.")
            return set()

        with zarrio_reader(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])
            unique_lead = np.unique(dummy.valid_time.data)
        return set(str(lt) for lt in unique_lead)

    def get_ensemble(self, stream: str | None = None) -> list[str]:
        """Get the list of ensemble member names for a given stream from the config.
        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            A list of ensemble members.
        """
        _logger.debug(f"Getting ensembles for stream {stream}...")

        if stream not in self._cached_ensemble:
            # TODO: improve this to get ensemble from io class
            with zarrio_reader(self.fname_zarr) as zio:
                dummy = zio.get_data(0, stream, zio.forecast_steps[0])
            self._cached_ensemble[stream] = list(dummy.prediction.as_xarray().coords["ens"].values)
        return self._cached_ensemble[stream]

    def is_gridded_data(self, stream: str) -> bool:
        """Check if the latitude and longitude coordinates are regularly spaced for a given stream.
        Parameters
        ----------
        stream :
            The name of the stream to get channels for.

        Returns
        -------
            True if the stream is regularly spaced. False otherwise.
        """
        if stream not in self._cached_is_gridded:
            self._cached_is_gridded[stream] = self._compute_is_gridded(stream)
        return self._cached_is_gridded[stream]

    def _compute_is_gridded(self, stream: str) -> bool:
        """Original is_gridded_data logic, called once per stream and cached."""
        _logger.debug(f"Checking regular spacing for stream {stream}...")

        with zarrio_reader(self.fname_zarr) as zio:
            dummy = zio.get_data(0, stream, zio.forecast_steps[0])

            sample_idx = zio.samples[1] if len(zio.samples) > 1 else zio.samples[0]
            fstep_idx = (
                zio.forecast_steps[1] if len(zio.forecast_steps) > 1 else zio.forecast_steps[0]
            )
            dummy1 = zio.get_data(sample_idx, stream, fstep_idx)

        da = dummy.prediction.as_xarray()
        da1 = dummy1.prediction.as_xarray()

        if (
            da["lat"].shape != da1["lat"].shape
            or da["lon"].shape != da1["lon"].shape
            or not (
                np.allclose(sorted(da["lat"].values), sorted(da1["lat"].values))
                and np.allclose(sorted(da["lon"].values), sorted(da1["lon"].values))
            )
        ):
            _logger.debug("Latitude and/or longitude coordinates are not regularly spaced.")
            return False
        else:
            _logger.debug("Latitude and longitude coordinates are regularly spaced.")
            return True


################### Helper functions ########################


def _select_channels(
    da_tar: xr.DataArray, da_pred: xr.DataArray, stream, channels, stream_cfg
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Preprocess the data by scaling z channels if needed and adding lead_time coordinate.

    Parameters
    ----------
    da_tar :
        Input DataArray to preprocess.
    da_pred :
        Input DataArray to preprocess.
    stream:
        Stream name, used to determine if z channels need to be scaled.
    channels:
        List of channels to select.
    stream_cfg:
        Stream configuration dictionary, used to determine if derived channels need to be computed.
    Returns
    -------
        Data arrays with selected channels and added derived channels if applicable.
    """
    # Ensure channel is a dimension, not a scalar coordinate (can happen after squeeze)
    if "channel" not in da_tar.dims:
        da_tar = da_tar.expand_dims("channel")
    if "channel" not in da_pred.dims:
        da_pred = da_pred.expand_dims("channel")

    assert da_pred.channel.values.tolist() == da_tar.channel.values.tolist(), (
        "Channels in prediction and target do not match."
    )

    all_channels = da_tar.channel.values.tolist()

    if set(channels) != set(all_channels):
        _logger.debug(
            f"Restricting targets and predictions to channels {channels} for stream {stream}..."
        )

        dc = DeriveChannels(
            all_channels,
            channels,
            stream_cfg,
        )

        da_tar, da_pred, channels = dc.get_derived_channels(da_tar, da_pred)

        # Verify that requested channels are available
        all_channels = da_tar.channel.values.tolist()
        missing_channels = set(channels) - set(all_channels)
        if missing_channels:
            _logger.warning(
                f"Skipping channels {missing_channels} for stream {stream}. "
                f"Not found in available channels."
            )
            channels = [ch for ch in channels if ch in all_channels]

        da_tar = da_tar.sel(channel=channels)
        da_pred = da_pred.sel(channel=channels)

    return da_tar, da_pred


def _scale_z_channels(data: xr.DataArray, stream: str) -> xr.DataArray:
    """
    Check scale all channels.

    Parameters
    ----------
    data :
        Input dataset
    stream :
        Stream name.
    Returns
    -------
        Returns a Dataset where channels have been scaled if needed
    """
    if stream is None or not str(stream).startswith("ERA5"):
        return data

    channels_z = [ch for ch in np.atleast_1d(data.channel.values) if str(ch).startswith("z_")]
    factor = 9.80665

    if channels_z:
        channels = data.channel.astype(str)
        mask = channels.str.startswith("z_")
        data = data.where(~mask, data / factor)
    return data


def _split_by_valid_time(arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    """
    Split arrays by valid_time and stack by sample, creating separate
    arrays for each unique lead_time.

    Lead_time is calculated as: valid_time - source_interval_start

    Parameters
    ----------
    arrays : list[xr.DataArray]
        List of DataArrays, each containing multiple valid_times per sample

    Returns
    -------
    list[xr.DataArray]
        List of DataArrays, one per unique lead_time, with samples
        stacked along 'sample' dimension
    """
    # Pre-compute all lead times and build index in single pass
    lead_time_groups = {}  # lead_time -> list of (arr_idx, ipoint_indices)

    unique_valid_times = [np.unique(da.valid_time.values) for da in arrays]

    if len(unique_valid_times) == len(arrays) and all(len(uvt) == 1 for uvt in unique_valid_times):
        _logger.debug(
            "All arrays have a single unique valid_time. Skipping splitting by valid_time."
        )
        arrays = _force_consistent_grids(arrays)

        return [arrays]

    for arr_idx, da in tqdm(enumerate(arrays), total=len(arrays), desc="Splitting by valid time"):
        vt = da.valid_time.values
        sis = da.source_interval_start.values

        # Calculate lead_time once
        if vt.ndim > 1:
            lead_times = vt - (sis[:, np.newaxis] if sis.ndim == 1 else sis)
            # Flatten and get unique lead times with their ipoint indices
            valid_mask = ~np.isnat(lead_times)
            for i in range(lead_times.shape[0]):
                row_leads = lead_times[i][valid_mask[i]]
                row_ipoints = np.where(valid_mask[i])[0]
                for lead, ipoint in zip(row_leads, row_ipoints, strict=False):
                    lead_time_groups.setdefault(lead, []).append((arr_idx, i, ipoint))
        else:
            lead_times = vt - sis
            valid_mask = ~np.isnat(lead_times)
            valid_leads = lead_times[valid_mask]
            valid_ipoints = np.where(valid_mask)[0]
            for lead, ipoint in zip(valid_leads, valid_ipoints, strict=False):
                lead_time_groups.setdefault(lead, []).append((arr_idx, 0, ipoint))

    # Get reference grid from first array for alignment
    ref_lat = arrays[0].lat.values
    ref_lon = arrays[0].lon.values
    ref_sort_idx = np.lexsort((ref_lon, ref_lat))
    ref_lat_sorted = ref_lat[ref_sort_idx]
    ref_lon_sorted = ref_lon[ref_sort_idx]

    # Process each lead time
    sorted_leads = sorted(lead_time_groups.keys())
    out = []

    for forecast_step, lead in enumerate(sorted_leads, start=1):
        # Group by array index to minimize selections
        array_groups = {}
        for arr_idx, sample_idx, ipoint in lead_time_groups[lead]:
            array_groups.setdefault(arr_idx, {}).setdefault(sample_idx, []).append(ipoint)

        per_sample = []
        for arr_idx, sample_dict in array_groups.items():
            da = arrays[arr_idx]

            for sample_idx, ipoint_list in sample_dict.items():
                # Single selection operation
                ipoint_arr = np.array(ipoint_list)
                da_subset = da.isel(ipoint=ipoint_arr)

                # Align to reference grid
                sort_idx = np.lexsort((da_subset.lon.values, da_subset.lat.values))
                da_subset = da_subset.isel(ipoint=sort_idx).assign_coords(
                    ipoint=np.arange(len(ipoint_arr)),
                    lat=("ipoint", ref_lat_sorted[: len(ipoint_arr)]),
                    lon=("ipoint", ref_lon_sorted[: len(ipoint_arr)]),
                )

                # Ensure sample dimension
                if "sample" not in da_subset.dims:
                    sample_val = da.sample.values.item() if da.sample.ndim == 0 else sample_idx
                    da_subset = da_subset.expand_dims(sample=[sample_val])

                per_sample.append(da_subset)

        if per_sample:
            # Single concat operation
            combined = xr.concat(per_sample, dim="sample", coords="different", compat="equals")
            combined = combined.assign_coords(
                ipoint=np.arange(combined.sizes["ipoint"]), forecast_step=forecast_step
            )
            out.append(combined)

    return out


def _add_lead_time_coord(da: xr.DataArray, sample_dim="sample") -> xr.DataArray:
    """
    Add lead_time coordinate computed as:
    valid_time - source_interval_start

    lead_time has dims (sample, ipoint) and dtype timedelta64[ns].

    Parameters
    ----------
    da :
        Input DataArray
    sample_dim :
        The name of the sample dimension (default is "sample") which should be kept.
        Collapse over the others.
    Returns
    -------
        Returns a DataArray with the lead_time coordinate added.

    NB. Need to be used AFTER splitting by valid_time and stacking by sample,
    so that all valid_times within a sample are the same and we can assign a
    single lead_time per sample.

    """
    vt = da["valid_time"].values
    sis = da["source_interval_start"].values
    # Compute lead_time: valid_time - source_interval_start

    if vt.ndim > 1:
        sis_expanded = sis[:, np.newaxis] if sis.ndim == 1 else sis
        lead_time_values = vt - sis_expanded
        # Get unique lead_time per sample, verify consistency
        lead_times = [
            np.unique(lead_time_values[i][~np.isnat(lead_time_values[i])])
            for i in range(lead_time_values.shape[0])
        ]
        if any(len(lt) != 1 for lt in lead_times):
            raise ValueError(
                "Inconsistent lead_time values within samples for "
                f"forecast_step {da.forecast_step.values}"
            )
        lead_time_per_sample = np.array([lt[0] for lt in lead_times])
    else:
        lead_time_values = vt - sis
        lead_time_per_sample = np.unique(lead_time_values[~np.isnat(lead_time_values)])

    # Verify all samples have same lead_time for this forecast_step
    unique_lead = np.unique(lead_time_per_sample)
    if len(unique_lead) != 1:
        raise ValueError(
            "Multiple lead_time values across samples for "
            f"forecast_step {da.forecast_step.values}: {unique_lead}"
        )

    da = da.assign_coords(lead_time=unique_lead[0])
    return da


def _force_consistent_grids(ref: list[xr.DataArray]) -> xr.DataArray:
    """
    Force all samples to share the same ipoint order.

    This function aligns the spatial ordering (lat/lon/ipoint) of all samples
    to that of the first sample, ensuring consistent spatial coordinates for
    subsequent concatenation. It is essential for regular-grid (gridded) data
    where spatial order matters but may differ across samples.

    Parameters
    ----------
    ref: list[xr.DataArray]
        List of xarray DataArrays, each representing one sample. Must have at least one element.

    Returns
    -------
    xr.DataArray
        A concatenated DataArray across the 'sample' dimension, where each sample's ipoint indices
        have been reordered to match the sorted lat/lon order of the first sample.

    Notes
    -----
    - All input DataArrays must share identical lat/lon values
        (though possibly in different orders).
    - Enforces consistent ipoint indexing after alignment (0..N-1).
    - Preserves and aligns all other coordinates and data variables.
    """
    assert len(ref) > 0, "_force_consistent_grids requires at least one input DataArray in 'ref'."

    # Pick first sample as reference
    ref_lat = ref[0].lat
    ref_lon = ref[0].lon

    sort_idx = np.lexsort((ref_lon.values, ref_lat.values))
    npoints = sort_idx.size
    aligned = []
    samples = []
    for i, a in enumerate(ref):
        a_sorted = a.isel(ipoint=sort_idx)
        samples.append(a_sorted.sample.values)
        a_sorted = a_sorted.assign_coords(
            ipoint=np.arange(npoints),
            lat=("ipoint", ref_lat.values[sort_idx]),
            lon=("ipoint", ref_lon.values[sort_idx]),
        )

        if "sample" not in a_sorted.dims:
            a_sorted = a_sorted.expand_dims(sample=[i])

        aligned.append(a_sorted)

    return xr.concat(aligned, dim="sample", coords="different", compat="equals")
