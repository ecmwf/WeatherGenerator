# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from weathergen.model.engines import (
    BilinearDecoder,
    EnsPredictionHead,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import is_stream_forcing

logger = logging.getLogger(__name__)


@dataclass
class PhysicalPrediction:
    """Decoder values and their corresponding rows in the native stream targets."""

    values: torch.Tensor
    indices: torch.Tensor


class DecoderModule(torch.nn.Module):
    """Physical readout for the streams selected by one named decoder."""

    def __init__(self, cf, targets_num_channels, targets_coords_size):
        super().__init__()
        self.cf = cf
        self.streams = cf.streams
        self.level = cf.healpix_level
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        for stream_name, si in self.streams.items():
            # skip decoder if channels are empty
            if is_stream_forcing(si):
                continue

            # skip for the moment to ensure target embedding and tte exist (ordering of
            # cf.streams is random)
            if si.get("pred_spatial_shared") is None:
                # extract and setup relevant parameters
                etc = si["embed_target_coords"]
                tr = si["target_readout"]
                num_layers = tr["num_layers"]
                tr_mlp_hidden_factor = tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
                tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
                softcap = tr["softcap"] if "softcap" in tr else 0.0

                dims_embed = [si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)]

                if is_root():
                    logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

                dim_coord_in = targets_coords_size[stream_name]

                # embedding network for coordinates
                if etc["net"] == "linear":
                    self.embed_target_coords[stream_name] = NamedLinear(
                        f"embed_target_coords_{stream_name}",
                        in_features=dim_coord_in,
                        out_features=dims_embed[0],
                        bias=False,
                    )
                elif etc["net"] == "mlp":
                    self.embed_target_coords[stream_name] = MLP(
                        dim_coord_in,
                        dims_embed[0],
                        hidden_factor=8,
                        with_residual=False,
                        dropout_rate=dropout_rate,
                        norm_eps=self.cf.mlp_norm_eps,
                        name=f"embed_target_coords_{stream_name}",
                    )
                else:
                    assert False

                if cf.decoder_type == "Linear":
                    tte = BilinearDecoder(
                        stream_name,
                        dims_embed[0],
                        cf.ae_global_dim_embed,
                        targets_num_channels[stream_name],
                    )
                else:
                    # target prediction engines
                    tte_version = (
                        TargetPredictionEngine
                        if cf.decoder_type != "PerceiverIOCoordConditioning"
                        else TargetPredictionEngineClassic
                    )
                    tte = tte_version(
                        cf,
                        dims_embed,
                        dim_coord_in,
                        tr_dim_head_proj,
                        tr_mlp_hidden_factor,
                        softcap,
                        stream_config=si,
                    )

                self.target_token_engines[stream_name] = tte

                # ensemble prediction heads to provide probabilistic prediction
                final_activation = si["pred_head"].get("final_activation", "Identity")
                if is_root():
                    logger.debug(
                        f"{final_activation} activation of pred head of {si['name']} stream"
                    )
                self.pred_heads[stream_name] = EnsPredictionHead(
                    dims_embed[-1],
                    targets_num_channels[stream_name],
                    si["pred_head"]["num_layers"],
                    si["pred_head"]["ens_size"],
                    norm_type=cf.norm_type,
                    final_activation=final_activation,
                    stream_name=stream_name,
                )

        # iterate again to setup shared spatial pred heads if specified in config
        for stream_name, si in self.streams.items():
            # skip decoder if channels are empty
            if is_stream_forcing(si):
                continue

            pred_spatial_shared = si.get("pred_spatial_shared")
            if pred_spatial_shared is not None:
                if pred_spatial_shared not in self.streams.keys():
                    msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                    msg += " but no stream with that name found."
                    raise ValueError(msg)
                if pred_spatial_shared == stream_name:
                    msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                    msg += "but cannot share with itself."
                    raise ValueError(msg)
                logger.debug(
                    f"{stream_name} shares spatial prediction head with {pred_spatial_shared}."
                )

                self.embed_target_coords[stream_name] = self.embed_target_coords[
                    pred_spatial_shared
                ]
                self.target_token_engines[stream_name] = self.target_token_engines[
                    pred_spatial_shared
                ]

                assert pred_spatial_shared in self.streams.keys()
                si_other = self.streams[pred_spatial_shared]
                dims_embed = [si_other["embed_target_coords"]["dim_embed"]]

                # ensemble prediction heads to provide probabilistic prediction
                final_activation = si["pred_head"].get("final_activation", "Identity")
                if is_root():
                    logger.debug(
                        f"{final_activation} activation of pred head of {si['name']} stream"
                    )
                self.pred_heads[stream_name] = EnsPredictionHead(
                    dims_embed[-1],
                    targets_num_channels[stream_name],
                    si["pred_head"]["num_layers"],
                    si["pred_head"]["ens_size"],
                    norm_type=cf.norm_type,
                    final_activation=final_activation,
                    stream_name=stream_name,
                )

    def forward(self, model_params, name, step, cells, batch):
        predictions = {}
        batch_size = len(batch)
        section = model_params.level_slices[self.level]
        if self.cf.decoder_type != "Linear":
            indices = getattr(model_params, f"decode_{name}_indices")
            # Context contains only this decoder grid's cells and their 1-ring.
            neighbors = cells[:, indices].flatten(0, -2)
            lengths = getattr(model_params, f"decode_{name}_lengths").repeat(batch_size)
            lengths = torch.cat([lengths.new_zeros(1), lengths])

        for stream_name in self.pred_heads:
            targets = [
                sample.streams_data[stream_name].decoder_targets[name][step]
                for sample in batch.samples
            ]
            t_coords_lens = [len(target.coords) for target in targets]
            t_coords = torch.cat([target.coords for target in targets])
            if len(t_coords) == 0:
                continue

            # embed token coords
            tc_embed = self.embed_target_coords[stream_name]
            tc_tokens = checkpoint(tc_embed, t_coords, use_reentrant=False)

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                logger.warning(
                    (
                        f"Skipping prediction for {stream_name} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                pred = torch.tensor([], device=tc_tokens.device)

            # skip empty lengths
            elif tc_tokens.shape[0] == 0:
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # lens for varlen attention
                tcls = torch.cat([target.lengths for target in targets])
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens,
                        cells[:, section].mean(dim=2).reshape(-1, cells.shape[-1]),
                        tcs_lens,
                    ).unsqueeze(0)  # add ensemble dim: shape is then [1, preds_per_coord, channels]
                else:
                    tc_tokens = self.target_token_engines[stream_name](
                        latent=neighbors,
                        output=tc_tokens,
                        latent_lens=lengths,
                        output_lens=tcs_lens,
                        coordinates=t_coords,
                    )

                    # final prediction head to map back to physical space
                    pred = self.pred_heads[stream_name](tc_tokens)

            # recover batch dimension (ragged, so as list)
            pred = torch.split(pred, t_coords_lens, dim=1)
            predictions[stream_name] = tuple(
                PhysicalPrediction(values, target.indices)
                for values, target in zip(pred, targets, strict=True)
            )

        return predictions
