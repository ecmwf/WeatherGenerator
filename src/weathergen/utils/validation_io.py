# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)


def write_output(
    cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out
):
    """Write model output with configurable step chunking.

    Unifies streaming and non-streaming modes via output.fstep_chunk_size:
    - fstep_chunk_size: 1         → Stream after each step
    - fstep_chunk_size: N         → Accumulate N steps, then write
    - fstep_chunk_size: total_steps → Non-streaming (write all at once)
    - fstep_chunk_size: None (default) → No streaming (equivalent to total_steps)
    """

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for loss_name, loss_term in val_cfg.losses.items()
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out_physical = target_aux_out[outputs_physical[0]]

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    total_steps = len(timestep_idxs)

    # Get chunking configuration (default: no streaming)
    fstep_chunk_size = val_cfg.get("output", {}).get("fstep_chunk_size", None)
    if fstep_chunk_size is None:
        fstep_chunk_size = total_steps

    # Collect source information (once, outside chunking loop)
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # Output stream configuration
    stream_names = [stream.name for stream in cf.streams]
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels = [list(stream.val_target_channels) for stream in cf.streams]
    source_channels = [list(stream.val_source_channels) for stream in cf.streams]
    geoinfo_channels = [[] for _ in cf.streams]

    sample_start = batch_idx * batch_size

    # Calculate source intervals
    start_date = val_cfg.start_date
    end_date = val_cfg.end_date
    twh = TimeWindowHandler(start_date, end_date, val_cfg.time_window_len, val_cfg.time_window_step)
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    # Write in chunks based on fstep_chunk_size
    fp32 = torch.float32
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        for chunk_start in range(0, total_steps, fstep_chunk_size):
            chunk_end = min(chunk_start + fstep_chunk_size, total_steps)
            chunk_indices = timestep_idxs[chunk_start:chunk_end]

            # Process and write chunk
            preds_chunk, targets_chunk, targets_coords_chunk, targets_times_chunk = [], [], [], []
            targets_lens_chunk = []

            for t_idx in chunk_indices:
                preds_step = []
                targets_step = []
                targets_coords_step = []
                targets_times_step = []
                targets_lens_step = []

                for stream_info in cf.streams:
                    sname = stream_info["name"]

                    if target_aux_out_physical.physical[t_idx][sname]["is_spoof"][0]:
                        preds = model_output.get_physical_prediction(t_idx, sname)
                        preds_shape = preds[0].shape if preds else (1, 1, 1)
                        preds_s = [
                            np.zeros((preds_shape[0], 0, preds_shape[2]))
                            for _ in range(len(preds) if preds else 1)
                        ]
                        targets_s = [
                            np.zeros((0, preds_shape[2])) for _ in range(len(preds) if preds else 1)
                        ]
                        t_coords_s = [np.zeros((0, 2)) for _ in range(len(preds) if preds else 1)]
                        t_times_s = [
                            np.array([]).astype("datetime64[ns]")
                            for _ in range(len(preds) if preds else 1)
                        ]
                    else:
                        preds = model_output.get_physical_prediction(t_idx, sname)
                        targets = target_aux_out_physical.physical[t_idx][sname]["target"]

                        preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

                        if preds is None:
                            assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                            preds = [target.clone().unsqueeze(0) for target in targets]

                        for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                            preds_s += [dn_data(sname, pred).detach().to(fp32).cpu().numpy()]
                            targets_s += [dn_data(sname, target).detach().to(fp32).cpu().numpy()]
                            target_data = target_aux_out_physical.physical[t_idx][sname]
                            t_coords_s += [target_data["target_coords"][i_batch].cpu().numpy()]
                            t_times_s += [
                                target_data["target_times"][i_batch].astype("datetime64[ns]")
                            ]

                    targets_lens_step += [t.shape[0] for t in targets_s]
                    preds_step += [np.concatenate(preds_s, axis=1)]
                    targets_step += [np.concatenate(targets_s)]
                    targets_coords_step += [np.concatenate(t_coords_s)]
                    targets_times_step += [np.concatenate(t_times_s)]

                if len(preds_step) == 0 or np.array([p.shape[1] for p in preds_step]).sum() == 0:
                    _logger.warning(f"Empty predictions for step {t_idx}")
                    continue

                preds_chunk.append(preds_step)
                targets_chunk.append(targets_step)
                targets_coords_chunk.append(targets_coords_step)
                targets_times_chunk.append(targets_times_step)
                targets_lens_chunk.append([targets_lens_step])

            if len(preds_chunk) == 0:
                continue

            # Determine forecast offset for this chunk
            forecast_offset = chunk_indices[0]

            data = io.OutputBatchData(
                sources,
                source_intervals,
                preds_chunk,
                targets_chunk,
                targets_coords_chunk,
                targets_times_chunk,
                targets_lens_chunk,
                output_streams,
                target_channels,
                source_channels,
                geoinfo_channels,
                sample_start,
                forecast_offset,
            )
            for subset in data.items():
                zio.write_zarr(subset)
