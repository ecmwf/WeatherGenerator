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


def _prepare_batch_data(cf, val_cfg, batch_size, batch_idx, batch):
    """Prepare common data independent of forecast steps."""
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

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

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date
    twh = TimeWindowHandler(start_date, end_date, val_cfg.time_window_len, val_cfg.time_window_step)
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    return (
        sources,
        source_intervals,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        sample_start,
    )


def _process_timestep(t_idx, dn_data, model_output, target_aux_out, cf):
    """Process a single forecast step and return data for writing."""
    fp32 = torch.float32
    preds_step = []
    targets_step = []
    targets_coords_step = []
    targets_times_step = []
    targets_lens_step = []

    for stream_info in cf.streams:
        sname = stream_info["name"]

        if target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
            preds = model_output.get_physical_prediction(t_idx, sname)
            preds_shape = preds[0].shape if preds else (1, 1, 1)
            preds_s = [np.zeros((preds_shape[0], 0, preds_shape[2])) for _ in range(len(preds) if preds else 1)]
            targets_s = [np.zeros((0, preds_shape[2])) for _ in range(len(preds) if preds else 1)]
            t_coords_s = [np.zeros((0, 2)) for _ in range(len(preds) if preds else 1)]
            t_times_s = [np.array([]).astype("datetime64[ns]") for _ in range(len(preds) if preds else 1)]
        else:
            preds = model_output.get_physical_prediction(t_idx, sname)
            targets = target_aux_out.physical[t_idx][sname]["target"]

            preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

            if preds is None:
                assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                preds = [target.clone().unsqueeze(0) for target in targets]

            for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                preds_s += [dn_data(sname, pred).detach().to(fp32).cpu().numpy()]
                targets_s += [dn_data(sname, target).detach().to(fp32).cpu().numpy()]
                target_data = target_aux_out.physical[t_idx][sname]
                t_coords_s += [target_data["target_coords"][i_batch].cpu().numpy()]
                t_times_s += [target_data["target_times"][i_batch].astype("datetime64[ns]")]

        targets_lens_step += [t.shape[0] for t in targets_s]
        preds_step += [np.concatenate(preds_s, axis=1)]
        targets_step += [np.concatenate(targets_s)]
        targets_coords_step += [np.concatenate(t_coords_s)]
        targets_times_step += [np.concatenate(t_times_s)]

    return preds_step, targets_step, targets_coords_step, targets_times_step, targets_lens_step


class StreamingOutputWriter:
    """Manages streaming output writing with callback mechanism."""

    def __init__(self, cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out):
        self.cf = cf
        self.val_cfg = val_cfg
        self.dn_data = dn_data
        self.model_output = model_output
        
        # Extract physical target_aux
        outputs_physical = [
            loss_name for loss_name, loss_term in val_cfg.losses.items() 
            if loss_term.type == "LossPhysical"
        ]
        assert len(outputs_physical) == 1
        self.target_aux_out = target_aux_out[outputs_physical[0]]
        
        # Prepare batch data once
        (
            self.sources,
            self.source_intervals,
            self.output_streams,
            self.target_channels,
            self.source_channels,
            self.geoinfo_channels,
            self.sample_start,
        ) = _prepare_batch_data(cf, val_cfg, batch_size, batch_idx, batch)
        
        # Streaming config
        self.streaming_cfg = val_cfg.get("output", {}).get("streaming", {})
        self.write_freq = self.streaming_cfg.get("num_steps", 1) if self.streaming_cfg.get("num_steps") else 1
        
        # Accumulate steps before writing
        self.accumulated_steps = []
        self._zarrio_writer = None

    def create_callback(self):
        """Create and return the step callback for model.forward()."""
        def callback(step, output):
            self.accumulated_steps.append(step)
            # Write when we have accumulated enough steps
            if len(self.accumulated_steps) >= self.write_freq:
                self._write_accumulated_steps()
                self.accumulated_steps = []
        return callback

    def _write_accumulated_steps(self):
        """Write the accumulated steps to zarr."""
        if self._zarrio_writer is None:
            self._zarrio_writer = zarrio_writer(config.get_path_results(self.cf, self.val_cfg.mini_epoch)).__enter__()

        for t_idx in self.accumulated_steps:
            preds_step, targets_step, targets_coords_step, targets_times_step, targets_lens_step = (
                _process_timestep(t_idx, self.dn_data, self.model_output, self.target_aux_out, self.cf)
            )

            if len(preds_step) == 0 or np.array([p.shape[1] for p in preds_step]).sum() == 0:
                _logger.warning(f"Empty predictions for step {t_idx}")
                continue

            data = io.OutputBatchData(
                self.sources,
                self.source_intervals,
                [preds_step],
                [targets_step],
                [targets_coords_step],
                [targets_times_step],
                [[targets_lens_step]],
                self.output_streams,
                self.target_channels,
                self.source_channels,
                self.geoinfo_channels,
                self.sample_start,
                t_idx,
            )
            for subset in data.items():
                self._zarrio_writer.write_zarr(subset)

    def flush(self):
        """Write any remaining accumulated steps and close."""
        if self.accumulated_steps:
            self._write_accumulated_steps()
        if self._zarrio_writer is not None:
            self._zarrio_writer.__exit__(None, None, None)


def write_output(
    cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, model_output, target_aux_out
):
    """Write model output with optional streaming based on config."""
    
    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name for loss_name, loss_term in val_cfg.losses.items() 
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out_physical = target_aux_out[outputs_physical[0]]

    # Check if streaming is enabled
    streaming_cfg = val_cfg.get("output", {}).get("streaming", {})
    streaming_enabled = streaming_cfg.get("enabled", False)

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    
    if streaming_enabled:
        # Create streaming writer and return callback for use in model.forward()
        writer = StreamingOutputWriter(
            cf, val_cfg, batch_size, mini_epoch, batch_idx, dn_data, batch, 
            model_output, target_aux_out
        )
        # Update mini_epoch in val_cfg (needed for zarr path)
        val_cfg.mini_epoch = mini_epoch
        
        # Return callback for model.forward() to use
        return writer.create_callback(), writer
    else:
        # Standard non-streaming: process all steps and write together
        forecast_offset = timestep_idxs[0]
        preds_all, targets_all, targets_coords_all, targets_times_all, targets_lens_all = [], [], [], [], []

        for t_idx in timestep_idxs:
            preds_step, targets_step, targets_coords_step, targets_times_step, targets_lens_step = (
                _process_timestep(t_idx, dn_data, model_output, target_aux_out_physical, cf)
            )

            if len(preds_step) == 0 or np.array([p.shape[1] for p in preds_step]).sum() == 0:
                _logger.warning(f"Empty predictions for step {t_idx}")
                continue

            preds_all.append(preds_step)
            targets_all.append(targets_step)
            targets_coords_all.append(targets_coords_step)
            targets_times_all.append(targets_times_step)
            targets_lens_all.append([targets_lens_step])

        if len(preds_all) > 0:
            (
                sources,
                source_intervals,
                output_streams,
                target_channels,
                source_channels,
                geoinfo_channels,
                sample_start,
            ) = _prepare_batch_data(cf, val_cfg, batch_size, batch_idx, batch)

            data = io.OutputBatchData(
                sources,
                source_intervals,
                preds_all,
                targets_all,
                targets_coords_all,
                targets_times_all,
                targets_lens_all,
                output_streams,
                target_channels,
                source_channels,
                geoinfo_channels,
                sample_start,
                forecast_offset,
            )
            with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
                for subset in data.items():
                    zio.write_zarr(subset)
        
        return None, None
