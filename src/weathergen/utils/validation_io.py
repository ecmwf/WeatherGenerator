# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import torch

import weathergen.common.config as config
import weathergen.common.io as io
import numpy as np
from weathergen.train.structures import TrainerPredictions
_logger = logging.getLogger(__name__)


def write_output(
    cf,
    epoch: int,
    batch_idx,
    sources: list[list[io.IOReaderData]],
    preds_all,
    targets_all,
    targets_coords_all,
    targets_times_all,
    targets_lens,
    preds: TrainerPredictions
):
    output_stream_names = cf.analysis_streams_output
    output_stream_names = ["ERA5", "tokens"]  # TEMPORARY OVERRIDE FOR TESTING
    export_tokens = "tokens" in output_stream_names
    stream_names = [stream.name for stream in cf.streams] + (["tokens"] if export_tokens else [])
    if output_stream_names is None:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}

    _logger.info(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in cf.streams]
    if export_tokens:
        token_channels = [f"tokens_{i}" for i in range(preds.tokens_all[0].shape[-1])]
        target_channels.append(token_channels)
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in cf.streams]
    # No source channels for tokens

    geoinfo_channels = [[] for _ in cf.streams]  # TODO obtain channels
    if export_tokens:
        geoinfo_channels.append([])

    # assume: is batch size guarnteed and constant:
    # => calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * cf.batch_size_validation_per_gpu

    if export_tokens:
        num_tokens = preds.tokens_all[0].shape[1]
        # Append the tokens to the output
        # Add a source for the tokens
        # TODO: is this correct? It should not
        iord = io.IOReaderData(
            coords=preds.tokens_coords_raw[0],
            geoinfos=np.array([]),  # No geoinfo for tokens
            data=preds.tokens_all[0].numpy(),
            datetimes=np.array([]),  # No times for tokens
        )
        sources = [fc_sources + [iord] for fc_sources in sources]
        # TODO: not sure if needed, it is not a target.
        for fc_target, fc_token in zip(targets_all, preds.tokens_all):
            fc_target.append([fc_token.squeeze().unsqueeze(-1)])  
        for fc_preds, fc_token in zip(preds_all, preds.tokens_all):
            fc_preds.append([fc_token.squeeze().unsqueeze(0)])  
        for fc_coords_target, fc_coords_token in zip(targets_coords_all, preds.tokens_coords_raw):
            fc_coords_target.append(torch.tensor(fc_coords_token))
        for fc_lens in targets_lens:
            fc_lens.append([num_tokens])
        # TODO: correct target times
        for fc_times_target in targets_times_all:
            fc_times_target.append(fc_times_target[0])  # Just repeating the existing ones
            



    extra_dim = 1 if export_tokens else 0
    assert len(stream_names) == len(targets_all[0]), f"data does not match number of streams, {len(stream_names),extra_dim,len(targets_all[0])}"
    assert len(stream_names) == len(preds_all[0]), f"data does not match number of streams, {len(stream_names),extra_dim,len(preds_all[0])}"
    assert len(stream_names) == len(sources[0]), f"data does not match number of streams, {len(stream_names),extra_dim,len(sources[0])}"


    data = io.OutputBatchData(
        sources,
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
        cf.forecast_offset,
    )

    with io.ZarrIO(config.get_path_output(cf, epoch)) as writer:
        for subset in data.items():
            writer.write_zarr(subset)
