# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from math import exp
import re

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)

# TODO: REMOVE LATER. ONLY FOR SINGLE-SAMPLE OVERFITTING EXPERIMENTS.
i = 0


# TODO: REMOVE LATER. ONLY FOR SINGLE-SAMPLE OVERFITTING EXPERIMENTS.
def _normalize_channel_name(name: str) -> str:
    return str(name).lower().replace("_", "").replace(" ", "")


# TODO: REMOVE LATER. ONLY FOR SINGLE-SAMPLE OVERFITTING EXPERIMENTS.
def _resolve_channel_names(stream_info, raw_channels):
    if not raw_channels:
        return raw_channels
    if isinstance(raw_channels[0], str):
        return list(raw_channels)

    channel_names = None
    if hasattr(stream_info, "val_target_channels") and stream_info.val_target_channels:
        if isinstance(stream_info.val_target_channels[0], str):
            channel_names = list(stream_info.val_target_channels)

    if channel_names is None:
        target_weights = getattr(stream_info, "target_channel_weights", None)
        if isinstance(target_weights, dict):
            channel_names = list(target_weights.keys())

    if channel_names is None:
        channel_weights = getattr(stream_info, "channel_weights", None)
        if isinstance(channel_weights, dict):
            channel_names = list(channel_weights.keys())

    if channel_names is None:
        return [f"ch{idx}" for idx in raw_channels]

    resolved = []
    for idx in raw_channels:
        if 0 <= int(idx) < len(channel_names):
            resolved.append(channel_names[int(idx)])
        else:
            resolved.append(f"ch{idx}")
    return resolved


def write_output(
    cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out, 
    noise_level=None,
    write_zarr=True,
):
    """
    Interface for writing model output

    Parameters
    ----------
    noise_level : float | None
        Fixed diffusion noise level (eta) used for this validation pass.
        When not None the value is embedded in plot filenames and titles.
    write_zarr : bool
        Whether to write zarr output. Default True. Set to False to only
        generate plots without writing zarr data.
    """
    # TODO: REMOVE LATER. ONLY FOR SINGLE-SAMPLE OVERFITTING EXPERIMENTS.
    global i

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    breakpoint()
    target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []
    noised_preds_all = []  # decoded noised tokens (diffusion models only)

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        noised_preds_all += [[]]
        targets_lens += [[]]
        for stream_idx, stream_info in enumerate(cf.streams):
            sname = stream_info["name"]

            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)

            if target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
                preds = model_output.get_physical_prediction(t_idx, sname)
                preds_shape = preds[0].shape
                # for-loop to make sure we have a consistent number of samples
                preds_s = [np.zeros((preds_shape[0], 0, preds_shape[2])) for _ in preds]
                targets_s = [np.zeros((0, preds_shape[2])) for _ in preds]
                t_coords_s = [np.zeros((0, 2)) for _ in preds]
                t_times_s = [np.array([]).astype("datetime64[ns]") for _ in preds]

            else:
                preds = model_output.get_physical_prediction(t_idx, sname)
                targets = target_aux_out.physical[t_idx][sname]["target"]
                
                preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

                # handle forcing streams or if sample is empty
                if preds is None:
                    # preds are empty so create copy of target and add ensemble dimension
                    assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                    preds = [target.clone().unsqueeze(0) for target in targets]

                for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                    target_data = target_aux_out.physical[t_idx][sname]
                    t_coords = target_data["target_coords"][i_batch]
                    t_times = target_data["target_times"][i_batch]

                    idxs_inv = target_aux_out.physical[t_idx][sname]["idxs_inv"][i_batch]
                    if idxs_inv is not None:
                        pred = pred[:, idxs_inv]
                        target = target[idxs_inv]
                        t_coords = t_coords[idxs_inv]
                        t_times = t_times[idxs_inv]

                    # denormalize data if requested and map to storage format
                    preds_s += [dn_data(sname, pred).detach().to(fp32).cpu().numpy()]
                    targets_s += [dn_data(sname, target).detach().to(fp32).cpu().numpy()]

                    # extract original target coords and times from target data
                    t_coords_s += [t_coords.cpu().numpy()]
                    t_times_s += [t_times.astype("datetime64[ns]")]
                    

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

            # collect decoded noised tokens (diffusion models only)
            noised_preds = model_output.get_noised_physical_prediction(t_idx, sname)
            if noised_preds is not None:
                noised_s = []
                for i_batch, npred in enumerate(noised_preds):
                    idxs_inv = target_aux_out.physical[t_idx][sname]["idxs_inv"][i_batch]
                    if idxs_inv is not None:
                        npred = npred[:, idxs_inv]
                    noised_s += [dn_data(sname, npred).detach().to(fp32).cpu().numpy()]
                noised_preds_all[-1] += [np.concatenate(noised_s, axis=1)]
            else:
                noised_preds_all[-1] += [np.array([])]

    if len(preds_all) == 0 or np.array([p.shape[1] for pp in preds_all for p in pp]).sum() == 0:
        _logger.warning("Writing no data since predictions are empty.")
        return

    # collect source information
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # more prep work

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = [stream.name for stream in cf.streams]
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in cf.streams]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in cf.streams]

    geoinfo_channels = [[] for _ in cf.streams]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    # write output

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date

    twh = TimeWindowHandler(
        start_date,
        end_date,
        val_cfg.time_window_len,
        val_cfg.time_window_step,
    )
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    data = io.OutputBatchData(
        sources,
        source_intervals,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        sample_start,
        forecast_offset,
    )
    if write_zarr:
        with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
            for subset in data.items():
                zio.write_zarr(subset)

    # Extract a representative date per batch sample from target times before
    # they are freed. Use the first non-NaT timestamp found in t_idx=0.
    sample_dates: list[str] = []
    if len(targets_times_all) > 0:
        for stream_times in targets_times_all[0]:
            if stream_times.size > 0:
                valid_times = stream_times[~np.isnat(stream_times)]
                if valid_times.size > 0:
                    sample_dates.append(str(valid_times[0].astype("datetime64[h]")))
                    break

    # Free arrays no longer needed after zarr writing
    del targets_all, targets_times_all, targets_lens, sources, data

    # TODO: REMOVE EVERYTHING BELOW THIS LINE LATER. ONLY FOR SINGLE-SAMPLE OVERFITTING EXPERIMENTS.

    # Prepare prediction data for plotting (scatter plot expects lat/lon coords on ipoint).
    base_plot_dir = config.get_path_run(cf) / "plots" / "validation"
    base_plot_dir.mkdir(parents=True, exist_ok=True)
    dpi_val = 150
    image_format = "png"
    # headline_channels = {"2t", "z500", "q850", "10u", "10v"}
    headline_channels = {"2t", "q850"}

    t_idx = np.random.randint(0, len(preds_all))  # TODO: loop over all time steps once plotting is set up
    for stream_idx, stream_info in enumerate(cf.streams):
        stream_name = stream_info["name"]
        preds_stream = preds_all[t_idx][stream_idx]
        coords_stream = targets_coords_all[t_idx][stream_idx]

        # Check for noised data for this stream
        noised_stream = noised_preds_all[t_idx][stream_idx]
        has_noised = noised_stream.size > 0 and noised_stream.ndim >= 2

        if preds_stream.size == 0 or coords_stream.size == 0:
            _logger.warning(f"No prediction data to plot for stream {stream_name}.")
            continue

        # Expected shape is (ens, ipoint, channel). Select first ensemble if present.
        if preds_stream.ndim == 3:
            preds_stream = preds_stream[0]
        elif preds_stream.ndim != 2:
            _logger.warning(
                f"Unsupported prediction shape {preds_stream.shape} for stream {stream_name}."
            )
            continue

        if has_noised:
            if noised_stream.ndim == 3:
                noised_stream = noised_stream[0]
            elif noised_stream.ndim != 2:
                has_noised = False

        channels = _resolve_channel_names(stream_info, target_channels[stream_idx])
        selected_channels = [
            ch for ch in channels if _normalize_channel_name(ch) in headline_channels
        ]
        if not selected_channels:
            _logger.warning(f"No headline channels available for plotting stream {stream_name}.")
            del preds_stream, coords_stream
            continue

        # Build a channel index map so we can slice numpy arrays directly
        # instead of constructing a full xarray DataArray for all channels.
        ch_to_col = {ch: idx for idx, ch in enumerate(channels)}

        lat = coords_stream[:, 0]
        lon = coords_stream[:, 1]

        run_id = config.get_run_id_from_config(cf)

        num_samples = len(preds)
        len_per_sample = preds_stream.shape[0] // num_samples
        noised_len_per_sample = noised_stream.shape[0] // num_samples if has_noised else 0

        if noise_level is not None:
            eta_str = re.sub(r'e[+]?0*(?=\d)', 'e', re.sub(r'e-0*(?=\d)', 'e-', f'{noise_level:.0e}'))
        else:
            eta_str = None
        eta_tag = f"_eta{eta_str}" if eta_str is not None else ""

        for sample in range(num_samples):
            s_start = sample * len_per_sample
            s_end = (sample + 1) * len_per_sample
            ns_start = sample * noised_len_per_sample if has_noised else 0
            ns_end = (sample + 1) * noised_len_per_sample if has_noised else 0

            # Extract sample date from target times
            sample_date_str = sample_dates[0] if len(sample_dates) > 0 else ""

            for varname in selected_channels:
                col = ch_to_col[varname]

                # --- denoised data ---
                den_vals = preds_stream[s_start:s_end, col]
                den_lat = lat[s_start:s_end]
                den_lon = lon[s_start:s_end]
                den_valid = ~np.isnan(den_vals)
                den_vals, den_lat, den_lon = den_vals[den_valid], den_lat[den_valid], den_lon[den_valid]

                # --- noised data ---
                if has_noised:
                    noi_vals = noised_stream[ns_start:ns_end, col]
                    noi_lat = lat[ns_start:ns_end]
                    noi_lon = lon[ns_start:ns_end]
                    noi_valid = ~np.isnan(noi_vals)
                    noi_vals, noi_lat, noi_lon = noi_vals[noi_valid], noi_lat[noi_valid], noi_lon[noi_valid]
                else:
                    noi_vals = noi_lat = noi_lon = None

                # Shared colour scale across both panels
                all_vals = np.concatenate([den_vals] + ([noi_vals] if noi_vals is not None else []))
                vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap("coolwarm")

                ncols = 2 if has_noised else 1
                proj = ccrs.Robinson()
                fig, axes = plt.subplots(
                    1, ncols, figsize=(8 * ncols, 5), dpi=dpi_val,
                    subplot_kw={"projection": proj},
                )
                if ncols == 1:
                    axes = [axes]

                # Left panel: noised (or skip if not available)
                if has_noised:
                    ax_noised = axes[0]
                    ax_noised.coastlines()
                    ax_noised.set_global()
                    sc_n = ax_noised.scatter(
                        noi_lon, noi_lat, c=noi_vals, norm=norm, cmap=cmap,
                        s=2.0, marker="o", transform=ccrs.PlateCarree(), linewidths=0.0,
                    )
                    if eta_str is not None:
                        ax_noised.set_title(f"Noised | {varname} | eta={eta_str}", fontsize=9.5)
                    else:
                        ax_noised.set_title(f"Noised | {varname}", fontsize=9.5)
                    ax_noised.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)
                    ax_denoised = axes[1]
                else:
                    ax_denoised = axes[0]

                # Right panel (or only panel): denoised
                ax_denoised.coastlines()
                ax_denoised.set_global()
                sc_d = ax_denoised.scatter(
                    den_lon, den_lat, c=den_vals, norm=norm, cmap=cmap,
                    s=2.0, marker="o", transform=ccrs.PlateCarree(), linewidths=0.0,
                )
                if eta_str is not None:
                    ax_denoised.set_title(f"Denoised | {varname} | eta={eta_str}", fontsize=9.5)
                else:
                    ax_denoised.set_title(f"Denoised | {varname}", fontsize=9.5)
                ax_denoised.gridlines(draw_labels=False, linestyle="--", color="black", linewidth=1)

                # Shared colourbar
                fig.colorbar(
                    sc_d, ax=axes, orientation="horizontal",
                    label=f"Variable: {varname}", fraction=0.05, pad=0.07,
                )

                date_part = f" | {sample_date_str}" if sample_date_str else ""
                if eta_str is not None:
                    fig.suptitle(
                        f"{stream_name} - {varname} (fstep {forecast_offset}) | sample {sample + 1}{date_part} | eta={eta_str}",
                        fontsize=11,
                    )
                else:
                    fig.suptitle(
                        f"{stream_name} - {varname} (fstep {forecast_offset}) | sample {sample + 1}{date_part}",
                        fontsize=11,
                    )

                channel_dir = base_plot_dir / varname
                channel_dir.mkdir(parents=True, exist_ok=True)
                epoch_tag = f"epoch_{mini_epoch:03d}_{i % 3}{eta_tag}"
                fname = channel_dir / f"{epoch_tag}.{image_format}"
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)

                del den_vals, den_lat, den_lon, den_valid
                if has_noised:
                    del noi_vals, noi_lat, noi_lon, noi_valid

        del preds_stream, coords_stream
        if has_noised:
            del noised_stream

    i += 1
