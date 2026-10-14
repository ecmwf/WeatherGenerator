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
from weathergen.datasets.domain import Domain
from weathergen.datasets.domain_pyramid import build_domain_pyramid
from weathergen.datasets.stream_levels import assign_stream_levels
from weathergen.model.engines import (
    EmbeddingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    Local2GlobalSumEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
)

# from weathergen.model.model import ModelParams
from weathergen.model.latent_cascade import LatentCascade
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

        # latent levels (one level unless `latent_levels` is configured). The single-level
        # attributes and modules below belong to the finest level; coarser levels of a
        # multi-resolution latent get their own modules in the per-level dicts.
        self.domain_pyramid = build_domain_pyramid(cf)
        self.multi_level = not self.domain_pyramid.is_single
        self.stream_encode_level, self.stream_decode_levels = assign_stream_levels(
            cf, ladder=self.domain_pyramid.levels
        )
        self.healpix_level = self.domain_pyramid.finest
        # healpix cells of the (possibly regional) domain
        self.domain: Domain = self.domain_pyramid.domain(self.healpix_level)
        self.num_healpix_cells = len(self.domain)

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
        self.q_cells = torch.nn.Parameter(self._build_q_cells(cf, self.domain), requires_grad=True)

        # query aggregation engine
        self.ae_aggregation_engine = QueryAggregationEngine(cf, self.num_healpix_cells)

        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

        # multi-resolution latent: modules of the coarser levels, keyed by str(level).
        # Embedding engine, local assimilation engine and local->global adapter are shared
        # across levels unless `share_local_assimilation: false`.
        self.latent_cascade: LatentCascade | None = None
        if self.multi_level:
            coarser = [lvl for lvl in self.domain_pyramid.levels if lvl != self.healpix_level]
            self.share_local_assimilation = cf.get("share_local_assimilation", True)
            if not self.share_local_assimilation:
                self._ae_local_per_level = torch.nn.ModuleDict()
                self._ae_local_global_per_level = torch.nn.ModuleDict()
                for lvl in coarser:
                    self._ae_local_per_level[str(lvl)] = LocalAssimilationEngine(cf)
                    self._ae_local_global_per_level[str(lvl)] = (
                        Local2GlobalSumEngine(cf)
                        if ae_adapter_type == "sum"
                        else Local2GlobalAssimilationEngine(cf)
                    )
            self._q_cells_param_dict = torch.nn.ParameterDict()
            self._ae_aggregation_per_level = torch.nn.ModuleDict()
            self._ae_global_per_level = torch.nn.ModuleDict()
            for lvl in coarser:
                dom_l = self.domain_pyramid.domain(lvl)
                self._q_cells_param_dict[str(lvl)] = torch.nn.Parameter(
                    self._build_q_cells(cf, dom_l), requires_grad=True
                )
                self._ae_aggregation_per_level[str(lvl)] = QueryAggregationEngine(cf, len(dom_l))
                self._ae_global_per_level[str(lvl)] = GlobalAssimilationEngine(cf, len(dom_l))

            # coarse <-> fine exchange after encoding (inert with num_cycles: 0)
            self.latent_cascade = LatentCascade.from_config(
                self.domain_pyramid, cf, "latent_cascade"
            )

        # optional: run the encoder without activation checkpointing (workaround for
        # recompute problems with FSDP2 DTensor parameters seen with multi-level runs)
        self._use_checkpoint = cf.get("encoder_use_checkpoint", True)

    def _build_q_cells(self, cf: Config, domain: Domain) -> torch.Tensor:
        """Initial learnable queries for the cells of `domain` (one latent level)."""
        if cf.ae_local_queries_per_cell:
            s = (len(domain), cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
            # add meta data
            # global nested ids of the cells (== arange(num_healpix_cells) for a global domain)
            cell_ids = torch.from_numpy(domain.active_cells)
            q_cells[:, :, -8:-6] = (
                (cell_ids / domain.num_total_cells)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 2))
            )
            theta, phi = healpy.pix2ang(nside=2**domain.healpix_level, ipix=cell_ids)
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
        return q_cells

    def _level_modules(self, level: int | None):
        """
        (q_cells, aggregation engine, global engine, local engine, local->global adapter)
        of a latent level; level None or the finest level -> the single-level modules.
        """
        if level is None or level == self.healpix_level:
            return (
                self.q_cells,
                self.ae_aggregation_engine,
                self.ae_global_engine,
                self.ae_local_engine,
                self.ae_local_global_engine,
            )
        key = str(level)
        if self.share_local_assimilation:
            ae_local, ae_local_global = self.ae_local_engine, self.ae_local_global_engine
        else:
            ae_local = self._ae_local_per_level[key]
            ae_local_global = self._ae_local_global_per_level[key]
        return (
            self._q_cells_param_dict[key],
            self._ae_aggregation_per_level[key],
            self._ae_global_per_level[key],
            ae_local,
            ae_local_global,
        )

    def initial_q_cells(self, state_dict_key: str) -> torch.Tensor | None:
        """
        Freshly initialised learnable queries for a model state-dict key, or None if the
        key is not a query bank. Used to fill query banks missing from a checkpoint
        (e.g. a latent level added when fine-tuning) on the FSDP loading path.
        """
        if state_dict_key == "encoder.q_cells":
            return self._build_q_cells(self.cf, self.domain).detach()
        prefix = "encoder._q_cells_param_dict."
        if self.multi_level and state_dict_key.startswith(prefix):
            lvl = int(state_dict_key[len(prefix) :])
            return self._build_q_cells(self.cf, self.domain_pyramid.domain(lvl)).detach()
        return None

    def reset_q_cells(self) -> None:
        """
        Re-draw the learnable cell queries of every latent level, with the same
        initialisation as in __init__.

        Needed after meta-device creation + to_empty() (the FSDP path): the model's
        reset_parameters() only re-initialises nn.Linear / nn.LayerNorm, so the queries
        would otherwise keep uninitialised memory (garbage or NaN).
        """
        from torch.distributed.tensor import DTensor, distribute_tensor

        banks = [(self.q_cells, self.domain)]
        if self.multi_level:
            banks += [
                (q, self.domain_pyramid.domain(int(lvl)))
                for lvl, q in self._q_cells_param_dict.items()
            ]
        with torch.no_grad():
            for q, domain in banks:
                full = self._build_q_cells(self.cf, domain).detach().to(dtype=q.dtype)
                if isinstance(q, DTensor):
                    full = full.to(q.to_local().device)
                    full = distribute_tensor(full, q.device_mesh, q.placements)
                else:
                    full = full.to(q.device)
                q.copy_(full)

    def q_cells_all_levels(self) -> list[torch.nn.Parameter]:
        """The learnable query banks of all latent levels."""
        q = [self.q_cells]
        if self.multi_level:
            q += list(self._q_cells_param_dict.values())
        return q

    def _checkpoint(self, fn, *args, **kwargs):
        if self._use_checkpoint:
            return checkpoint(fn, *args, **kwargs)
        kwargs.pop("use_reentrant", None)
        return fn(*args, **kwargs)

    def forward(self, model_params, batch):
        """
        Encoder forward

        Single latent level: returns (tokens_global, posteriors).
        Multi-resolution latent: returns ({level: tokens_global}, {level: posteriors}).
        """

        if self.multi_level:
            return self.forward_multi_level(model_params, batch)

        stream_cell_tokens = self._checkpoint(
            self.embed_engine, batch, model_params.pe_embed, use_reentrant=False
        )

        tokens_global, posteriors = self._checkpoint(
            self.assimilate_local, model_params, stream_cell_tokens, batch, use_reentrant=False
        )

        tokens_global = self._checkpoint(
            self.ae_global_engine,
            tokens_global,
            coords=model_params.rope_coords,
            use_reentrant=False,
        )

        return tokens_global, posteriors

    def forward_multi_level(self, model_params, batch):
        """
        Encoder forward for the multi-resolution latent.

        Every latent level runs the single-level pipeline on the streams encoded at that
        level (embedding -> local assimilation -> local->global adapter -> query
        aggregation -> global assimilation), with that level's queries, engines and model
        parameters (`model_params.params_for(level)`). The latent cascade then exchanges
        information between adjacent levels.
        """
        tokens_by_level, posteriors_by_level = {}, {}
        for lvl in self.domain_pyramid.levels:
            mp_l = model_params.params_for(lvl)
            ae_global_l = self._level_modules(lvl)[2]
            stream_names_l = [n for n in self.stream_names if self.stream_encode_level[n] == lvl]

            stream_cell_tokens = self._checkpoint(
                self.embed_engine,
                batch,
                mp_l.pe_embed,
                batch.tokens_lens[lvl],
                stream_names_l,
                use_reentrant=False,
            )

            tokens_global, posteriors = self._checkpoint(
                self.assimilate_local,
                mp_l,
                stream_cell_tokens,
                batch,
                lvl,
                use_reentrant=False,
            )

            tokens_global = self._checkpoint(
                ae_global_l,
                tokens_global,
                coords=mp_l.rope_coords,
                use_reentrant=False,
            )
            tokens_by_level[lvl] = tokens_global
            posteriors_by_level[lvl] = posteriors

        tokens_by_level = self.latent_cascade(tokens_by_level)

        return tokens_by_level, posteriors_by_level

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

    def assimilate_local_project_chunked(
        self, tokens, tokens_global, cell_lens, q_cells_lens, level=None
    ):
        """
        Apply the local assimilation engine and then the
        local-to-global adapter using a chunking in the number of tokens
        to work around to bug in flash attention, the computations is performed in chunks

        `level` selects the latent level (multi-resolution latent); None -> finest level.
        """

        _, _, _, ae_local_engine, ae_local_global_engine = self._level_modules(level)
        healpix_level = self.healpix_level if level is None else level
        num_healpix_cells = len(self.domain_pyramid.domain(healpix_level))

        # combined cell lens for all tokens in batch across all input steps
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)

        # subdivision factor for required splitting
        clen = num_healpix_cells // (2 if healpix_level <= 5 else 8)
        # a small regional domain can make clen 0, which would skip the loop below entirely
        clen = max(1, clen)
        tokens_global_unmasked = []
        posteriors = []

        # Multi-resolution latent under FSDP: a level can have empty chunks on some ranks
        # only (e.g. a regional level partly outside a stream's grid). Skipping them would
        # make the FSDP collectives of the engines below diverge across ranks (deadlock), so
        # an empty chunk then runs on a one-token dummy whose output is multiplied by zero.
        run_empty_chunks = (
            self.multi_level
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        dummy_sink = None

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
            is_empty = bool(l0 == l1 or toks.shape[0] == 0)
            if is_empty and not run_empty_chunks:
                continue

            if is_empty:
                # one-token dummy chunk (see run_empty_chunks above). The dummy input must
                # require grad: FSDP2 attaches its post-backward hook (reduce-scatter +
                # reshard) to a module call only if at least one input requires grad.
                # With a constant input the first local block's post-backward is deferred
                # to the end of backward, so this rank issues one all-gather less and in a
                # different order than ranks without empty chunks -> NCCL deadlock.
                toks = tokens.new_zeros(
                    (1, *tokens.shape[1:]), requires_grad=torch.is_grad_enabled()
                )
                toks_global = tokens_global[0:1]
                cell_lens_cur = torch.cat([zero_pad, torch.ones_like(zero_pad)])
            else:
                toks_global = tokens_global[i * clen : i_end]
                cell_lens_cur = torch.cat([zero_pad, cell_lens[i * clen : i_end]])
            q_cells_lens_cur = q_cells_lens[: cell_lens_cur.shape[0]]

            # local assimilation model
            toks = ae_local_engine(toks, cell_lens_cur, use_reentrant=False)

            toks, posteriors_c = self.interpolate_latents(toks)
            if not is_empty:
                posteriors += [posteriors_c]

            # create mask for global tokens, without first element (used for padding)
            mask = cell_lens_cur[1:].to(torch.bool)
            toks_global_unmasked = toks_global[mask]
            q_cells_lens_unmasked = torch.cat([zero_pad, q_cells_lens_cur[1:][mask]])
            cell_lens_unmasked = torch.cat([zero_pad, cell_lens_cur[1:][mask]])

            # local to global adapter engine
            toks_global_unmasked = ae_local_global_engine(
                toks,
                toks_global_unmasked,
                q_cells_lens_unmasked,
                cell_lens_unmasked,
            )

            if is_empty:
                # keep the dummy in the autograd graph (scaled to zero) so that the backward
                # pass also runs the same collectives on every rank
                zero = toks_global_unmasked.sum() * 0.0
                dummy_sink = zero if dummy_sink is None else dummy_sink + zero
                continue

            tokens_global_unmasked += [toks_global_unmasked]

        if len(tokens_global_unmasked) == 0:
            assert False, "Not yet implemented"
        tokens_global_unmasked = torch.cat(tokens_global_unmasked)
        if dummy_sink is not None:
            tokens_global_unmasked = tokens_global_unmasked + dummy_sink

        return tokens_global_unmasked, posteriors

    def aggregation_engine_unmasked(
        self,
        tokens_global_unmasked,
        tokens_global_register_class,
        tokens_lens,
        rope_cell_coords=None,
        ae_aggregation_engine=None,
    ):
        """
        Aggregation engine on the global latents of unmasked cells
        """

        if ae_aggregation_engine is None:
            ae_aggregation_engine = self.ae_aggregation_engine

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
        if rope_cell_coords is not None:
            num_extra = self.num_class_tokens + self.num_register_tokens
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
        tokens_global_unmasked = ae_aggregation_engine(
            tokens_global_unmasked, batch_lens_patched, use_reentrant=False, coords=packed_coords
        )

        return tokens_global_unmasked

    def assimilate_local(
        self, model_params, tokens: torch.Tensor, batch: ModelBatch, level: int | None = None
    ) -> torch.Tensor:
        """
        Processes embedded tokens locally and prepares them for the global assimilation

        Args:
            model_params : Query and embedding parameters
            tokens : Input tokens to be processed by local assimilation
            cell_lens : Used to identify range of tokens to use from generated tokens in cell
                embedding
            level : latent level (multi-resolution latent); None -> single/finest level,
                in which case `model_params` and `batch.tokens_lens` are used as they are
        Returns:
            Tokens for global assimilation
        """

        q_cells, ae_aggregation_engine, _, _, _ = self._level_modules(level)
        tokens_lens = batch.tokens_lens if level is None else batch.tokens_lens[level]
        num_healpix_cells = (
            self.num_healpix_cells if level is None else len(self.domain_pyramid.domain(level))
        )

        cell_lens = torch.sum(tokens_lens, 2).flatten()

        num_steps_input = batch.get_num_source_steps()
        rs = num_steps_input * len(batch)

        # create register and latent tokens and prepend to latent spatial tokens
        num_extra_tokens = self.num_register_tokens + self.num_class_tokens
        pos_enc = positional_encoding_harmonic
        tokens_global_register_class = pos_enc(q_cells.repeat(rs, num_extra_tokens, 1))

        # TODO: re-enable or remove ae_local_queries_per_cell
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (q_cells + model_params.pe_global).repeat(rs, 1, 1)
        else:
            num_tokens = num_healpix_cells
            tokens_global = q_cells.repeat(num_tokens, 1, 1) + model_params.pe_global
            tokens_global = tokens_global.repeat(rs, 1, 1)

        # apply local assimilation engine and project onto global latent vectors
        tokens_global_unmasked, posteriors = self.assimilate_local_project_chunked(
            tokens, tokens_global, cell_lens, model_params.q_cells_lens, level
        )

        # apply aggregation engine on unmasked tokens
        tokens_global_unmasked = self.aggregation_engine_unmasked(
            tokens_global_unmasked,
            tokens_global_register_class,
            tokens_lens,
            rope_cell_coords=model_params.rope_cell_coords,
            ae_aggregation_engine=ae_aggregation_engine,
        )

        # final processing

        tokens_global = (
            torch.permute(tokens_global, [1, 0, 2]).squeeze().reshape(rs, num_healpix_cells, -1)
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
        cell_lens_r = cell_lens.unsqueeze(0).reshape(rs, num_healpix_cells)
        mask = torch.cat([mask_reg_class_tokens, cell_lens_r.to(torch.bool)], dim=1)

        # fill empty tensor using mask for positions of unmasked tokens
        tokens_global[mask] = tokens_global_unmasked.to(tokens_global.dtype)

        # recover batch dimension and build global token list
        num_tokens_tot = num_healpix_cells + self.num_register_tokens + self.num_class_tokens
        q_c_shape = q_cells.shape
        tokens_global = (
            tokens_global.reshape([rs, num_tokens_tot, q_c_shape[-2], q_c_shape[-1]])
            #  removing this line because else they get added twice? + model_params.pe_global
        ).flatten(1, 2)

        return tokens_global, posteriors
