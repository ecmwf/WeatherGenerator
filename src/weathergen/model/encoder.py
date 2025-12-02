# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import astropy_healpix as hp
import astropy_healpix.healpy
from astropy_healpix import healpy

from weathergen.common.config import Config
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.blocks import CrossAttentionBlock, OriginalPredictionBlock, SelfAttentionBlock
from weathergen.model.embeddings import (
    StreamEmbedLinear,
    StreamEmbedTransformer,
)

from weathergen.model.engines import (
    EmbeddingEngine,
    LocalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    GlobalAssimilationEngine,
    )

# from weathergen.model.model import ModelParams
from weathergen.model.layers import MLP
from weathergen.model.utils import ActivationFactory
from weathergen.utils.utils import get_dtype

from weathergen.model.parametrised_prob_dist import LatentInterpolator


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
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size
        
        ##############
        # embedding engine
        # determine stream names once so downstream components use consistent keys
        self.stream_names = [str(stream_cfg["name"]) for stream_cfg in cf.streams]
        # separate embedding networks for differnt observation types
        self.embed_engine = EmbeddingEngine(cf, self.sources_size, self.stream_names)
        
        ##############
        # local assimilation engine
        self.ae_local_engine = LocalAssimilationEngine(cf)

        if self.cf.latent_noise_kl_weight > 0.0:
            self.interpolate_latents = LatentInterpolator(
                gamma=self.cf.latent_noise_gamma,
                dim=self.cf.ae_local_dim_embed,
                use_additive_noise=self.cf.latent_noise_use_additive_noise,
                deterministic=self.cf.latent_noise_deterministic_latents,
            )

        ##############
        # local -> global assimilation engine adapter
        self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)

        ##############
        # learnable queries
        if self.cf.ae_local_queries_per_cell:
            s = (self.num_healpix_cells, self.cf.ae_local_num_queries, self.cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / self.cf.ae_global_dim_embed
            # add meta data
            q_cells[:, :, -8:-6] = (
                (torch.arange(self.num_healpix_cells) / self.num_healpix_cells)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat((1, self.cf.ae_local_num_queries, 2))
            )
            theta, phi = healpy.pix2ang(
                nside=2**self.healpix_level, ipix=torch.arange(self.num_healpix_cells)
            )
            q_cells[:, :, -6:-3] = (
                torch.cos(theta).unsqueeze(1).unsqueeze(1).repeat((1, self.cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -3:] = (
                torch.sin(phi).unsqueeze(1).unsqueeze(1).repeat((1, self.cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -9] = torch.arange(self.cf.ae_local_num_queries)
            q_cells[:, :, -10] = torch.arange(self.cf.ae_local_num_queries)
        else:
            s = (1, self.cf.ae_local_num_queries, self.cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / self.cf.ae_global_dim_embed
        self.q_cells = torch.nn.Parameter(q_cells, requires_grad=True)

        ##############
        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

    def forward(self, model_params, streams_data, source_cell_lens):
        
        # embed
        tokens = self.embed_cells(model_params, streams_data, source_cell_lens)

        # local assimilation engine and adapter
        tokens, posteriors = self.assimilate_local(model_params, tokens, source_cell_lens)

        tokens = self.assimilate_global(tokens)

        return tokens, posteriors

    #########################################
    def embed_cells(
        self, model_params, streams_data, source_cell_lens
    ) -> torch.Tensor:
        """Embeds input data for each stream separately and rearranges it to cell-wise order
        Args:
            model_params : Query and embedding parameters
            streams_data : Used to initialize first tokens for pre-processing
        Returns:
            Tokens for local assimilation
        """

        device = next(self.parameters()).device
        tokens_all = self.embed_engine(
            streams_data, source_cell_lens, model_params.pe_embed, self.dtype, device
        )

        return tokens_all

    #########################################
    def assimilate_local(
        self, model_params, tokens: torch.Tensor, cell_lens: torch.Tensor
    ) -> torch.Tensor:
        """Processes embedded tokens locally and prepares them for the global assimilation
        Args:
            model_params : Query and embedding parameters
            tokens : Input tokens to be processed by local assimilation
            cell_lens : Used to identify range of tokens to use from generated tokens in cell
                embedding
        Returns:
            Tokens for global assimilation
        """

        batch_size = (
            self.cf.batch_size_per_gpu if self.training else self.cf.batch_size_validation_per_gpu
        )

        s = self.q_cells.shape
        # print( f'{np.prod(np.array(tokens.shape))} :: {np.prod(np.array(s))}'
        #        + ':: {np.prod(np.array(tokens.shape))/np.prod(np.array(s))}')
        # TODO: test if positional encoding is needed here
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (self.q_cells + model_params.pe_global).repeat(batch_size, 1, 1)
        else:
            tokens_global = (
                self.q_cells.repeat(self.num_healpix_cells, 1, 1) + model_params.pe_global
            )
        q_cells_lens = torch.cat(
            [model_params.q_cells_lens[0].unsqueeze(0)]
            + [model_params.q_cells_lens[1:] for _ in range(batch_size)]
        )

        # local assimilation model
        # for block in self.ae_local_blocks:
        #     tokens = checkpoint(block, tokens, cell_lens, use_reentrant=False)

        # if self.cf.latent_noise_kl_weight > 0.0:
        #     tokens, posteriors = self.interpolate_latents.interpolate_with_noise(
        #         tokens, sampling=self.training
        #     )
        # else:
        #     tokens, posteriors = tokens, 0.0

        # for block in self.ae_adapter:
        #     tokens_global = checkpoint(
        #         block,
        #         tokens_global,
        #         tokens,
        #         q_cells_lens,
        #         cell_lens,
        #         use_reentrant=False,
        #     )

        # work around to bug in flash attention for hl>=5

        istep = 0

        cell_lens = cell_lens[istep][1:]
        clen = self.num_healpix_cells // (2 if self.cf.healpix_level <= 5 else 8)
        tokens_global_all = []
        posteriors = []
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)
        for i in range((cell_lens.shape[0]) // clen):
            # make sure we properly catch all elements in last chunk
            i_end = (i + 1) * clen if i < (cell_lens.shape[0] // clen) - 1 else cell_lens.shape[0]
            l0, l1 = (
                (0 if i == 0 else cell_lens[: i * clen].cumsum(0)[-1]),
                cell_lens[:i_end].cumsum(0)[-1],
            )

            tokens_c = tokens[l0:l1]
            tokens_global_c = tokens_global[i * clen : i_end]
            cell_lens_c = torch.cat([zero_pad, cell_lens[i * clen : i_end]])
            q_cells_lens_c = q_cells_lens[: cell_lens_c.shape[0]]

            if l0 == l1 or tokens_c.shape[0] == 0:
                tokens_global_all += [tokens_global_c]
                continue

            # local assimilation model
            tokens_c = self.ae_local_engine(tokens_c, cell_lens_c, use_reentrant=False)

            if self.cf.latent_noise_kl_weight > 0.0:
                tokens_c, posteriors_c = self.interpolate_latents.interpolate_with_noise(
                    tokens_c, sampling=self.training
                )
                posteriors += [posteriors_c]
            else:
                tokens_c, posteriors = tokens_c, 0.0

            tokens_global_c = self.ae_local_global_engine(
                tokens_c, tokens_global_c, q_cells_lens_c, cell_lens_c, use_reentrant=False
            )

            tokens_global_all += [tokens_global_c]

        tokens_global = torch.cat(tokens_global_all)

        # recover batch dimension and build global token list
        tokens_global = (
            tokens_global.reshape([batch_size, self.num_healpix_cells, s[-2], s[-1]])
            + model_params.pe_global
        ).flatten(1, 2)

        return tokens_global, posteriors

    #########################################
    def assimilate_global(self, tokens: torch.Tensor) -> torch.Tensor:
        """Performs transformer based global assimilation in latent space
        Args:
            model_params : Query and embedding parameters (never used)
            tokens : Input tokens to be pre-processed by global assimilation
        Returns:
            Latent representation of the model
        """

        # global assimilation engine and adapter
        tokens = self.ae_global_engine(tokens, use_reentrant=False)

        return tokens