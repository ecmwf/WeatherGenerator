# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
import math

import numpy as np
import torch
from astropy_healpix import healpy
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.engines import (
    EmbeddingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    Local2GlobalSumEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
)

# from weathergen.model.model import ModelParams
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.positional_encoding import (
    build_spherical_rope_coeff_tensors,
    get_rope_mode,
    get_rope_spherical_band,
    positional_encoding_harmonic,
)
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


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

        self.dtype = get_dtype(cf.attention_dtype)

        # Positional embeddings
        self.max_tokens_local_per_cell = cf.get("ae_local_max_tokens_per_cell", 64)
        self.register_buffer(
            "pe_embed",
            torch.zeros(self.max_tokens_local_per_cell, cf.ae_local_dim_embed, dtype=self.dtype),
        )

        self.register_buffer(
            "q_cells_lens", torch.ones(self.num_healpix_cells + 1, dtype=torch.int32)
        )
        self.q_cells_lens[0] = 0

        self.register_buffer(
            "pe_global",
            torch.zeros(
                self.num_healpix_cells,
                cf.ae_local_num_queries,
                cf.ae_global_dim_embed,
                dtype=self.dtype,
            ),
        )

        # RoPE coordinates
        self.rope_mode = get_rope_mode(cf, logger)
        if self.rope_mode != "none":
            self.num_extra_tokens = cf.num_register_tokens + cf.num_class_tokens
            total_tokens = (
                self.num_healpix_cells + self.num_extra_tokens
            ) * cf.ae_local_num_queries
            self.register_buffer(
                "rope_coords",
                torch.zeros(
                    1,
                    total_tokens,
                    2,
                    dtype=self.dtype,
                ),
            )
            self.register_buffer(
                "rope_cell_coords",
                torch.zeros(
                    self.num_healpix_cells,
                    2,
                    dtype=self.dtype,
                ),
            )
            if self.rope_mode == "spherical":
                rope_spherical_band = get_rope_spherical_band(cf)
                num_modes = 2 * int(rope_spherical_band) + 1
                self.register_buffer(
                    "rope_spherical_coeffs",
                    torch.zeros(1, total_tokens, num_modes, 2, dtype=self.dtype),
                )
                self.register_buffer(
                    "rope_spherical_cell_coeffs",
                    torch.zeros(self.num_healpix_cells, num_modes, 2, dtype=self.dtype),
                )
                self.register_buffer(
                    "rope_spherical_extra_coeffs",
                    torch.zeros(self.num_extra_tokens, num_modes, 2, dtype=self.dtype),
                )
            else:
                self.rope_spherical_coeffs = None
                self.rope_spherical_cell_coeffs = None
                self.rope_spherical_extra_coeffs = None
        else:
            self.rope_coords = None
            self.rope_cell_coords = None
            self.rope_spherical_coeffs = None
            self.rope_spherical_cell_coeffs = None
            self.rope_spherical_extra_coeffs = None

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
        self.stream_names = list(cf.streams.keys())
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
        ae_adapter_type = cf.get("ae_adapter_type", "cross_attention")
        if ae_adapter_type == "sum":
            self.ae_local_global_engine = Local2GlobalSumEngine(cf)
        else:
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

    def reset_parameters(self) -> None:
        """Creates positional embedding for each grid point for each stream used after stream
        embedding, positional embedding for all stream assimilated cell-level local embedding,
        initializing queries for local-to-global adapters, HEALPix neighbourhood based parameter
        initializing for target prediction.

        Sinusoidal positional encoding: Harmonic positional encoding based upon sine and cosine for
        both per stream after stream embedding and per cell level for local assimilation.

        Query len based parameter creation: Calculate parameters for the calculated token length at
        each cell after local assimilation."""

        cf = self.cf

        dim_embed = cf.ae_local_dim_embed
        token_idx_bias = 16
        freq_bias = 8
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(
            token_idx_bias,
            token_idx_bias + self.max_tokens_local_per_cell,
            device=self.pe_embed.device,
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(self.max_tokens_local_per_cell) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_mode != "none":
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            self.rope_cell_coords.data.copy_(coords)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0)
            offset = self.num_extra_tokens * cf.ae_local_num_queries
            self.rope_coords.data.fill_(0.0)
            self.rope_coords.data[:, offset : offset + coords_flat.shape[1], :].copy_(coords_flat)

            if self.rope_mode == "spherical":
                band = int(get_rope_spherical_band(cf))
                (
                    (cell_real, cell_imag),
                    (extra_real, extra_imag),
                    (packed_extra_real, packed_extra_imag),
                    (packed_real, packed_imag),
                ) = build_spherical_rope_coeff_tensors(
                    nside=2**self.healpix_level,
                    band=band,
                    num_local_queries=cf.ae_local_num_queries,
                    num_extra_tokens=self.num_extra_tokens,
                    device=self.rope_spherical_coeffs.device,
                    dtype=self.rope_spherical_coeffs.dtype,
                )
                self.rope_spherical_cell_coeffs.data[..., 0].copy_(cell_real)
                self.rope_spherical_cell_coeffs.data[..., 1].copy_(cell_imag)
                self.rope_spherical_extra_coeffs.data[..., 0].copy_(extra_real)
                self.rope_spherical_extra_coeffs.data[..., 1].copy_(extra_imag)

                self.rope_spherical_coeffs.data.fill_(0.0)
                self.rope_spherical_coeffs.data[:, :offset, :, 0].copy_(packed_extra_real)
                self.rope_spherical_coeffs.data[:, :offset, :, 1].copy_(packed_extra_imag)
                self.rope_spherical_coeffs.data[
                    :, offset : offset + packed_real.shape[1], :, 0
                ].copy_(packed_real)
                self.rope_spherical_coeffs.data[
                    :, offset : offset + packed_imag.shape[1], :, 1
                ].copy_(packed_imag)

        self.pe_global.data.fill_(0.0)
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed
        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 0::2] += (
            torch.sin(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 1::2] += (
            torch.cos(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )

        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

    def forward(self, batch):
        """
        Encoder forward
        """

        stream_cell_tokens = checkpoint(
            self.embed_engine, batch, self.pe_embed, use_reentrant=False
        )

        tokens_global, posteriors = checkpoint(
            self.assimilate_local, stream_cell_tokens, batch, use_reentrant=False
        )

        tokens_global = checkpoint(
            self.ae_global_engine,
            tokens_global,
            coords=(
                self.rope_spherical_coeffs.unbind(dim=-1)
                if self.rope_spherical_coeffs is not None
                else self.rope_coords
            ),
            use_reentrant=False,
        )

        return tokens_global, posteriors

    def interpolate_latents(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        self,
        tokens_global_unmasked,
        tokens_global_register_class,
        tokens_lens,
        rope_cell_coords=None,
        rope_cell_coeffs=None,
        rope_extra_coeffs=None,
    ):
        """
        Aggregation engine on the global latents of unmasked cells
        """

        zero_pad = torch.zeros(1, device=tokens_global_unmasked.device, dtype=torch.int32)

        # permute to use ae_local_num_queries as the batchsize and no_of_tokens
        # as seq len for flash attention
        tokens_global_unmasked = torch.permute(tokens_global_unmasked, [1, 0, 2])

        cell_lens_unflattened = torch.sum(tokens_lens, 2)
        cell_mask = cell_lens_unflattened.to(torch.bool)
        batch_lens = cell_mask.sum(dim=-1).flatten()
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

        # Build packed coords matching the interleaved token order
        num_extra = self.num_class_tokens + self.num_register_tokens
        if rope_cell_coeffs is not None:
            extra_real, extra_imag = rope_extra_coeffs.unbind(dim=-1)
            cell_real, cell_imag = rope_cell_coeffs.unbind(dim=-1)
            packed_real = []
            packed_imag = []
            for mask_b in cell_mask.flatten(0, 1):
                packed_real.append(extra_real)
                packed_imag.append(extra_imag)
                packed_real.append(cell_real[mask_b])
                packed_imag.append(cell_imag[mask_b])
            packed_coords = (torch.cat(packed_real, dim=0), torch.cat(packed_imag, dim=0))
        elif rope_cell_coords is not None:
            zero_coords = torch.zeros(
                num_extra, 2, device=rope_cell_coords.device, dtype=rope_cell_coords.dtype
            )
            packed_coords = []
            for mask_b in cell_mask.flatten(0, 1):
                packed_coords.append(zero_coords)
                packed_coords.append(rope_cell_coords[mask_b])
            packed_coords = torch.cat(packed_coords, dim=0)
        else:
            packed_coords = None

        batch_lens = batch_lens + (self.num_class_tokens + self.num_register_tokens)
        batch_lens_patched = torch.cat([zero_pad, batch_lens], dim=0)
        tokens_global_unmasked = self.ae_aggregation_engine(
            tokens_global_unmasked, batch_lens_patched, use_reentrant=False, coords=packed_coords
        )

        return tokens_global_unmasked

    def assimilate_local(self, tokens: torch.Tensor, batch: ModelBatch) -> torch.Tensor:
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

        num_steps_input = batch.get_num_source_steps()
        rs = num_steps_input * len(batch)

        # create register and latent tokens and prepend to latent spatial tokens
        num_extra_tokens = self.num_register_tokens + self.num_class_tokens
        pos_enc = positional_encoding_harmonic
        tokens_global_register_class = pos_enc(self.q_cells.repeat(rs, num_extra_tokens, 1))

        # TODO: re-enable or remove ae_local_queries_per_cell
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (self.q_cells + self.pe_global).repeat(rs, 1, 1)
        else:
            num_tokens = self.num_healpix_cells
            tokens_global = self.q_cells.repeat(num_tokens, 1, 1) + self.pe_global
            tokens_global = tokens_global.repeat(rs, 1, 1)

        # apply local assimilation engine and project onto global latent vectors
        tokens_global_unmasked, posteriors = self.assimilate_local_project_chunked(
            tokens, tokens_global, cell_lens, self.q_cells_lens
        )

        # apply aggregation engine on unmasked tokens
        tokens_global_unmasked = self.aggregation_engine_unmasked(
            tokens_global_unmasked,
            tokens_global_register_class,
            batch.tokens_lens,
            rope_cell_coords=self.rope_cell_coords,
            rope_cell_coeffs=self.rope_spherical_cell_coeffs,
            rope_extra_coeffs=self.rope_spherical_extra_coeffs,
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
