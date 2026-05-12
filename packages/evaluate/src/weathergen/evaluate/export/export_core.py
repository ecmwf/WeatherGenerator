import logging
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import get_model_results
from weathergen.common.io import zarrio_reader
from weathergen.evaluate.export.parser_factory import CfParserFactory
from weathergen.evaluate.export.reshape import detect_grid_type

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Module-level cache for the zarr path and open store — resolved once per worker.
_CACHED_FNAME_ZARR: str | None = None
_CACHED_ZIO = None


def _init_worker(fname_zarr: str) -> None:
    """Pool initializer: open the zarr store once and keep it for the worker's lifetime."""
    global _CACHED_FNAME_ZARR, _CACHED_ZIO
    _CACHED_FNAME_ZARR = fname_zarr
    _CACHED_ZIO = zarrio_reader(fname_zarr)
    _CACHED_ZIO.__enter__()


def get_data_worker(args: tuple) -> tuple[int, int, xr.DataArray]:
    """
    Worker function to retrieve data for a single (sample, fstep) pair.

    Reads the raw zarr arrays as numpy (bypassing dask) and builds a
    lightweight xarray DataArray that can be pickled back to the main
    process with all data already in memory.

    Returns
    -------
        Tuple of (sample, fstep, xarray.DataArray) with data fully in memory.
    """
    sample, fstep, stream, dtype = args

    # Navigate directly to the zarr group for this (sample, stream, fstep, dtype).
    group_path = f"{sample}/{stream}/{fstep}/{dtype}"
    ds_group = _CACHED_ZIO.data_root.get(group_path)

    if ds_group is None:
        raise FileNotFoundError(f"Zarr group '{group_path}' not found in {_CACHED_FNAME_ZARR}")

    # Read raw arrays as numpy — no dask, no chunking overhead.
    data_arr = np.asarray(ds_group["data"])  # (npoints, nchannels) or (npoints, nchannels, nens)
    coords_arr = np.asarray(ds_group["coords"])  # (npoints, 2)
    times_arr = np.asarray(ds_group["times"]).astype("datetime64[ns]")  # (npoints,)
    channels = list(ds_group.attrs["channels"])

    # Build a lightweight xarray DataArray with the same structure
    # that process_sample / assign_coords expects:
    #   dims = [ipoint, channel]
    #   coords: forecast_step, channel, valid_time, lat, lon
    npoints = data_arr.shape[0]

    # Handle optional ensemble dimension: squeeze it out if present.
    if data_arr.ndim == 3 and data_arr.shape[2] == 1:
        data_arr = data_arr[:, :, 0]

    da_result = xr.DataArray(
        data_arr,
        dims=["ipoint", "channel"],
        coords={
            "ipoint": np.arange(npoints),
            "channel": channels,
            "forecast_step": fstep,
            "valid_time": ("ipoint", times_arr),
            "lat": ("ipoint", coords_arr[:, 0]),
            "lon": ("ipoint", coords_arr[:, 1]),
        },
    )

    return (sample, fstep, da_result)


def get_fsteps(fsteps, fname_zarr: str):
    """
    Retrieve available forecast steps from the Zarr store and filter
    based on requested forecast steps.

    Parameters
    ----------
        fsteps : list
            List of requested forecast steps.
            If None, retrieves all available forecast steps.
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[int]
            List of forecast steps to be used for data retrieval.
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])

    if fsteps is None:
        return zio_forecast_steps

    requested = sorted([int(fstep) for fstep in fsteps])
    available_set = set(zio_forecast_steps)
    valid = [f for f in requested if f in available_set]
    missing = [f for f in requested if f not in available_set]

    if missing:
        _logger.warning(
            f"Requested forecast steps {missing} are not available in the zarr store "
            f"(available: {zio_forecast_steps}). They will be skipped."
        )

    if not valid:
        raise ValueError(
            f"None of the requested forecast steps {requested} exist in the zarr store. "
            f"Available forecast steps: {zio_forecast_steps}"
        )

    return valid


def get_samples(samples, fname_zarr: str):
    """
    Retrieve available samples from the Zarr store
    and filter based on requested samples.
    Parameters
    ----------
        samples : list
            List of requested samples. If None, retrieves all available samples.
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[int]
            List of samples to be used for data retrieval.
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_samples = sorted([int(sample) for sample in zio.samples])

    if samples is None:
        return zio_samples

    requested = sorted([int(sample) for sample in samples])
    available_set = set(zio_samples)
    valid = [s for s in requested if s in available_set]
    missing = [s for s in requested if s not in available_set]

    if missing:
        _logger.warning(
            f"Requested samples {missing} are not available in the zarr store "
            f"(available range: {zio_samples[0]}–{zio_samples[-1]}). They will be skipped."
        )

    if not valid:
        raise ValueError(
            f"None of the requested samples {requested} exist in the zarr store. "
            f"Available samples: {zio_samples}"
        )

    return valid


def get_channels(channels, stream: str, fname_zarr: str) -> list[str]:
    """
    Retrieve available channels from the Zarr store and filter based on requested channels.
    Parameters
    ----------
        channels : list
            List of requested channels. If None, retrieves all available channels.
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[str]
            List of channels to be used for data retrieval.
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels

        if channels is not None:
            existing_channels = set(all_channels) & set(channels)
            if existing_channels != set(channels):
                missing_channels = set(channels) - set(existing_channels)
                _logger.warning(
                    "The following requested channels are"
                    f"not available in the data and will be skipped: {missing_channels}"
                )
        return all_channels if channels is None else list(existing_channels)


def get_grid_type(data_type, stream: str, fname_zarr: str) -> str:
    """
    Determine the grid type of the data (regular or gaussian).
    Parameters
    ----------
        data_type : str
            Type of data to retrieve ('target' or 'prediction').
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        str
            Grid type ('regular' or 'gaussian').
    """
    with zarrio_reader(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        data = dummy_out.target if data_type == "target" else dummy_out.prediction
        return detect_grid_type(data.as_xarray().squeeze())


# TODO: this will change after restructuring the lead time.
def get_source_info(fname_zarr, stream, samples) -> tuple[list[np.datetime64], list[np.datetime64]]:
    """
    Retrieve source interval boundaries from the source group at forecast step 0.

    Values are derived from the actual ``times`` array of the **source**
    group at forecast step 0:
    - ``source_start = min(source_times)``
    - ``source_end   = max(source_times)``

    The ``source_end`` also serves as the reference (initialisation) time.

    Parameters
    ----------
    fname_zarr : str
        Path to the Zarr store.
    stream : str
        Stream name to retrieve data for (e.g., 'ERA5').
    samples : list
        List of samples to process.

    Returns
    -------
    tuple[list, list]
        ``(source_starts, source_ends)`` — one entry per sample,
        all as ``datetime64[ns]``.
    """
    _logger.info(f"Retrieving source info for {len(samples)} samples...")

    source_starts = []
    source_ends = []
    with zarrio_reader(fname_zarr) as zio:
        for sample in tqdm(samples, desc="Getting source info"):
            group_path = f"{sample}/{stream}/0/source"
            source_group = zio.data_root.get(group_path)

            if source_group is None:
                raise FileNotFoundError(f"Zarr group '{group_path}' not found in {fname_zarr}")

            times_arr = np.asarray(source_group["times"]).astype("datetime64[ns]")
            source_start = np.min(times_arr)
            source_end = np.max(times_arr)

            _logger.debug(f"Sample {sample}: source_interval=[{source_start} .. {source_end}]")
            source_starts.append(source_start)
            source_ends.append(source_end)

    return source_starts, source_ends


def get_streams(stream, fname_zarr):
    with zarrio_reader(fname_zarr) as zio:
        zio_streams = zio.streams
    streams = zio_streams if stream is None else [stream]
    return streams


def export_model_outputs(data_type: str, config: OmegaConf, **kwargs) -> None:
    """
    Retrieve data from Zarr store and export to the requested format.

    All (sample, fstep) pairs are submitted to the pool at once so that
    every worker stays busy.  Results are grouped by sample and handed to
    the parser in sample order.

    Parameters
    ----------
    data_type: str
        Type of data to retrieve ('target' or 'prediction').
    config : OmegaConf
            Loaded config for cf_parser function.
    kwargs:
        Additional keyword arguments for the parser.
    """
    kwargs = OmegaConf.create(kwargs)

    run_id = kwargs.run_id
    samples = kwargs.samples
    fsteps = kwargs.fsteps
    stream = kwargs.stream
    channels = kwargs.channels
    n_processes = kwargs.n_processes
    epoch = kwargs.epoch
    rank = kwargs.rank

    if data_type not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {data_type}. Must be 'target' or 'prediction'.")

    fname_zarr = get_model_results(run_id, epoch, rank)
    fsteps = get_fsteps(fsteps, fname_zarr)
    samples = get_samples(samples, fname_zarr)
    streams = get_streams(stream, fname_zarr)
    for stream in streams:
        grid_type = get_grid_type(data_type, stream, fname_zarr)
        channels = get_channels(channels, stream, fname_zarr)
        source_starts, source_ends = get_source_info(fname_zarr, stream, samples)
        kwargs["grid_type"] = grid_type
        kwargs["channels"] = channels
        kwargs["data_type"] = data_type

        parser = CfParserFactory.get_parser(config=config, **kwargs)

        n_fsteps = len(fsteps)
        total_tasks = len(samples) * n_fsteps

        # Batch size in *samples*. Limits how many samples can be in-flight at once,
        # bounding peak memory while still allowing read/write overlap within each batch.
        batch_size = max(1, n_processes * 2)
        n_batches = (len(samples) + batch_size - 1) // batch_size

        _logger.info(
            f"Exporting {len(samples)} samples × {n_fsteps} fsteps "
            f"({total_tasks} total tasks) in {n_batches} batch(es) of up to "
            f"{batch_size} samples, using {n_processes} workers. "
            f"Reading and writing are interleaved within each batch."
        )

        # Initialise each worker with the zarr path so it is resolved only once.
        with Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(fname_zarr,),
        ) as pool:
            samples_written = 0

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(samples))
                batch_samples = samples[batch_start:batch_end]
                batch_source_starts = source_starts[batch_start:batch_end]
                batch_source_ends = source_ends[batch_start:batch_end]

                # Map sample -> index within this batch for ref_times lookup.
                sample_to_batch_idx = {s: i for i, s in enumerate(batch_samples)}

                batch_tasks = [
                    (sample, fstep, stream, data_type)
                    for sample in batch_samples
                    for fstep in fsteps
                ]

                _logger.info(
                    f"Batch {batch_idx + 1}/{n_batches}: "
                    f"samples {batch_start}–{batch_end - 1} "
                    f"({len(batch_samples)} samples, {len(batch_tasks)} tasks)"
                )

                # Interleaved read/write: as soon as all fsteps for a sample
                # arrive, write it immediately while workers continue reading.
                sample_results: dict[int, list] = defaultdict(list)
                batch_written = 0

                pbar = tqdm(
                    total=len(batch_tasks),
                    desc=f"  Batch {batch_idx + 1}/{n_batches}",
                )

                processed_samples = []

                for sample, _fstep, data in pool.imap_unordered(
                    get_data_worker, batch_tasks, chunksize=1
                ):
                    sample_results[sample].append(data)
                    pbar.update(1)

                    # Check if this sample is complete (all fsteps received).
                    if len(sample_results[sample]) == n_fsteps:
                        b_idx = sample_to_batch_idx[sample]
                        source_start = batch_source_starts[b_idx]
                        source_end = batch_source_ends[b_idx]
                        results_iter = iter(sample_results[sample])
                        processed = parser.process_sample(
                            results_iter,
                            ref_time=source_end,
                            source_interval_start=source_start,
                            source_interval_end=source_end,
                        )
                        processed_samples.append(processed)

                        # Free memory immediately.
                        del sample_results[sample]
                        batch_written += 1

                # Only save here if need to merge samples, otherwise saved in process_sample
                if processed_samples[0] is not None:
                    parser.save(processed_samples)
                pbar.close()

                samples_written += batch_written
                if batch_written != len(batch_samples):
                    _logger.error(
                        f"Batch {batch_idx + 1}: expected {len(batch_samples)} "
                        f"samples but only wrote {batch_written}. "
                        f"Incomplete: {list(sample_results.keys())}"
                    )

                # Free any remaining refs before next batch.
                del sample_results

        _logger.info(f"Export complete. Wrote {samples_written}/{len(samples)} samples.")
