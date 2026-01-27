# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from astropy_healpix import healpy
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.model.engines import (
    EmbeddingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
)

# from weathergen.model.model import ModelParams
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.positional_encoding import positional_encoding_harmonic


class EncoderModule(torch.nn.Module):
    name: "EncoderModule"

    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        :param stream_names: Ordered list of stream identifiers aligned with cf.streams.
        """
        super(EncoderModule, self).__init__()
        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.ae_aggregation_engine: QueryAggregationEngine | None = None
        self.ae_global_engine: GlobalAssimilationEngine | None = None
        self.ae_local_engine: LocalAssimilationEngine | None = None
        self.ae_local_global_engine: Local2GlobalAssimilationEngine | None = None
        self.embed_engine: EmbeddingEngine | None = None
        self.interpolator_latents: LatentInterpolator | None = None

        # embedding engine
        # determine stream names once so downstream components use consistent keys
        self.stream_names = [str(stream_cfg["name"]) for stream_cfg in cf.streams]
        # separate embedding networks for differnt observation types
        self.embed_engine = EmbeddingEngine(cf, self.sources_size)

        assert cf.ae_global_att_dense_rate == 1.0, "Local attention not adapted for register tokens"
        self.num_register_tokens = cf.num_register_tokens
        self.num_class_tokens = cf.num_class_tokens

        # local assimilation engine
        self.ae_local_engine = LocalAssimilationEngine(cf)

        if cf.latent_noise_kl_weight > 0.0:
            self.interpolator_latents = LatentInterpolator(
                gamma=cf.latent_noise_gamma,
                dim=cf.ae_local_dim_embed,
                use_additive_noise=cf.latent_noise_use_additive_noise,
                deterministic=cf.latent_noise_deterministic_latents,
            )

        # local -> global assimilation engine adapter
        self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)

        # learnable queries
        if cf.ae_local_queries_per_cell:
            s = (self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
            # add meta data
            q_cells[:, :, -8:-6] = (
                (torch.arange(self.num_healpix_cells) / self.num_healpix_cells)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 2))
            )
            theta, phi = healpy.pix2ang(
                nside=2**self.healpix_level, ipix=torch.arange(self.num_healpix_cells)
            )
            q_cells[:, :, -6:-3] = (
                torch.cos(theta).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -3:] = (
                torch.sin(phi).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -9] = torch.arange(cf.ae_local_num_queries)
            q_cells[:, :, -10] = torch.arange(cf.ae_local_num_queries)
        else:
            s = (1, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
        self.q_cells = torch.nn.Parameter(q_cells, requires_grad=True)

        # query aggregation engine
        self.ae_aggregation_engine = QueryAggregationEngine(cf, self.num_healpix_cells)

        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

    def forward(self, model_params, batch):
        """
        Encoder forward
        """

        stream_cell_tokens = checkpoint(
            self.embed_engine, batch, model_params.pe_embed, use_reentrant=False
        )

        tokens_global, posteriors = checkpoint(
            self.assimilate_local, model_params, stream_cell_tokens, batch, use_reentrant=False
        )

        tokens_global = checkpoint(self.ae_global_engine, tokens_global, use_reentrant=False)

        return tokens_global, posteriors

    def interpolate_latents(self, tokens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ "
        TODO
        """

        if self.cf.latent_noise_kl_weight > 0.0:
            tokens, posteriors = self.interpolator_latents.interpolate_with_noise(
                tokens, sampling=self.stage
            )
        else:
            posteriors = torch.zeros((1,), device=tokens.device)

        return tokens, posteriors

    def assimilate_local_project_chunked(self, tokens, tokens_global, cell_lens, q_cells_lens):
        """
        Apply the local assimilation engine and then the
        local-to-global adapter using a chunking in the number of tokens
        to work around to bug in flash attention, the computations is performed in chunks
        """

        # combined cell lens for all tokens in batch across all input steps
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)

        # subdivision factor for required splitting
        clen = self.num_healpix_cells // (2 if self.cf.healpix_level <= 5 else 8)
        tokens_global_unmasked = []
        posteriors = []

        for i in range(cell_lens.shape[0] // clen):
            # make sure we properly catch all elements in last chunk
            i_end = (i + 1) * clen if i < (cell_lens.shape[0] // clen) - 1 else cell_lens.shape[0]
            l0, l1 = (
                (0 if i == 0 else cell_lens[: i * clen].cumsum(0)[-1]),
                cell_lens[:i_end].cumsum(0)[-1],
            )

            toks = tokens[l0:l1]
            # if we have a very sparse input, we may have no tokens in the chunk, toks
            # skip processing of the empty chunk in this case
            # Check if this chunk is empty
            if l0 == l1 or toks.shape[0] == 0:
                continue

            toks_global = tokens_global[i * clen : i_end]
            cell_lens_cur = torch.cat([zero_pad, cell_lens[i * clen : i_end]])
            q_cells_lens_cur = q_cells_lens[: cell_lens_cur.shape[0]]

            # local assimilation model
            toks = self.ae_local_engine(toks, cell_lens_cur, use_reentrant=False)

            toks, posteriors_c = self.interpolate_latents(toks)
            posteriors += [posteriors_c]

            # create mask for global tokens, without first element (used for padding)
            mask = cell_lens_cur[1:].to(torch.bool)
            toks_global_unmasked = toks_global[mask]
            q_cells_lens_unmasked = torch.cat([zero_pad, q_cells_lens_cur[1:][mask]])
            cell_lens_unmasked = torch.cat([zero_pad, cell_lens_cur[1:][mask]])

            # local to global adapter engine
            toks_global_unmasked = self.ae_local_global_engine(
                toks,
                toks_global_unmasked,
                q_cells_lens_unmasked,
                cell_lens_unmasked,
            )

            tokens_global_unmasked += [toks_global_unmasked]

        if len(tokens_global_unmasked) == 0:
            assert False, "Not yet implemented"
        tokens_global_unmasked = torch.cat(tokens_global_unmasked)

        return tokens_global_unmasked, posteriors

    def aggregation_engine_unmasked(
        self, tokens_global_unmasked, tokens_global_register_class, tokens_lens
    ):
        """
        Aggregation engine on the global latents of unmasked cells
        """

        zero_pad = torch.zeros(1, device=tokens_global_unmasked.device, dtype=torch.int32)

        # permute to use ae_local_num_queries as the batchsize and no_of_tokens
        # as seq len for flash attention
        tokens_global_unmasked = torch.permute(tokens_global_unmasked, [1, 0, 2])

        cell_lens_unflattened = torch.sum(tokens_lens, 2)
        batch_lens = cell_lens_unflattened.to(torch.bool).sum(dim=-1).flatten()
        expected_len = batch_lens.sum().item()
        actual_len = tokens_global_unmasked.shape[1]
        assert expected_len == actual_len, (
            f"Shape mismatch: expected {expected_len}, got {actual_len}"
        )
        tokens_global_unmasked = torch.split(tokens_global_unmasked.squeeze(0), list(batch_lens))
        tokens_global_unmasked = torch.cat(
            [
                t
                for tup in zip(tokens_global_register_class, tokens_global_unmasked, strict=False)
                for t in tup
            ],
            dim=0,
        )

        batch_lens = batch_lens + (self.num_class_tokens + self.num_register_tokens)
        batch_lens_patched = torch.cat([zero_pad, batch_lens], dim=0)
        tokens_global_unmasked = self.ae_aggregation_engine(
            tokens_global_unmasked, batch_lens_patched, use_reentrant=False
        )

        return tokens_global_unmasked

    def assimilate_local(
        self, model_params, tokens: torch.Tensor, batch: ModelBatch
    ) -> torch.Tensor:
        """
        Processes embedded tokens locally and prepares them for the global assimilation

        Args:
            model_params : Query and embedding parameters
            tokens : Input tokens to be processed by local assimilation
            cell_lens : Used to identify range of tokens to use from generated tokens in cell
                embedding
        Returns:
            Tokens for global assimilation
        """

        cell_lens = torch.sum(batch.tokens_lens, 2).flatten()

        num_steps_input = batch.get_num_steps()
        rs = num_steps_input * len(batch)

        # create register and latent tokens and prepend to latent spatial tokens
        num_extra_tokens = self.num_register_tokens + self.num_class_tokens
        pos_enc = positional_encoding_harmonic
        tokens_global_register_class = pos_enc(self.q_cells.repeat(rs, num_extra_tokens, 1))

        # TODO: re-enable or remove ae_local_queries_per_cell
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (self.q_cells + model_params.pe_global).repeat(rs, 1, 1)
        else:
            num_tokens = self.num_healpix_cells
            tokens_global = self.q_cells.repeat(num_tokens, 1, 1) + model_params.pe_global
            tokens_global = tokens_global.repeat(rs, 1, 1)

        # apply local assimilation engine and project onto global latent vectors
        tokens_global_unmasked, posteriors = self.assimilate_local_project_chunked(
            tokens, tokens_global, cell_lens, model_params.q_cells_lens
        )

        # apply aggregation engine on unmasked tokens
        tokens_global_unmasked = self.aggregation_engine_unmasked(
            tokens_global_unmasked, tokens_global_register_class, batch.tokens_lens
        )

        # final processing

        tokens_global = (
            torch.permute(tokens_global, [1, 0, 2])
            .squeeze()
            .reshape(rs, self.num_healpix_cells, -1)
        )
        # TODO, TODO, TODO: do we need this
        tokens_global = torch.cat([tokens_global_register_class, tokens_global], dim=1)

        # create mask from cell lens
        mask_reg_class_tokens = (
            torch.ones(
                self.num_register_tokens + self.num_class_tokens,
                device=tokens_global.device,
            )
            .to(torch.bool)
            .unsqueeze(0)
            .repeat(rs, 1)
        )
        cell_lens_r = cell_lens.unsqueeze(0).reshape(rs, self.num_healpix_cells)
        mask = torch.cat([mask_reg_class_tokens, cell_lens_r.to(torch.bool)], dim=1)

        # fill empty tensor using mask for positions of unmasked tokens
        tokens_global[mask] = tokens_global_unmasked.to(tokens_global.dtype)

        # recover batch dimension and build global token list
        num_tokens_tot = self.num_healpix_cells + self.num_register_tokens + self.num_class_tokens
        q_c_shape = self.q_cells.shape
        tokens_global = (
            tokens_global.reshape([rs, num_tokens_tot, q_c_shape[-2], q_c_shape[-1]])
            #  removing this line because else they get added twice? + model_params.pe_global
        ).flatten(1, 2)

        return tokens_global, posteriors
