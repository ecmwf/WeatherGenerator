# DOCS-model.md — WeatherGenerator model architecture

One-line purpose: `src/weathergen/model/` implements the WeatherGenerator network — a hierarchical
transformer that assimilates arbitrary multi-stream Earth-system observations into a latent state on
a HEALPix grid, rolls that state forward in time, and decodes it at arbitrary target coordinates.

Part of the agent docs rooted at `WeatherGenerator/AGENT-README.md`. Training-side view (losses,
EMA teacher usage, trainer loop): `src/weathergen/train/DOCS-Train.md`.

All paths below are relative to `src/weathergen/model/`. Line numbers are approximate anchors, not
exact references — re-grep if they have drifted.

---

## 1. Data flow at a glance

```
ModelBatch (per-stream tokenized sources, per-cell token counts in batch.tokens_lens)
  │
  ├─ EncoderModule (encoder.py)
  │    EmbeddingEngine          per-stream nets -> tokens in ae_local_dim_embed, cell-ordered
  │    LocalAssimilationEngine  varlen self-attention *within* each HEALPix cell
  │    (LatentInterpolator)     optional VAE-style noise/KL on local latents
  │    Local2Global adapter     cross-attention: learnable per-cell queries <- local tokens
  │    QueryAggregationEngine   varlen self-attention over unmasked cells (+ register/class tokens)
  │    scatter into full grid   masked cells keep query+pe_global content
  │    GlobalAssimilationEngine dense/local self-attention over all cells
  │
  ▼
latent tokens: (batch, num_aux_tokens + num_healpix_cells * ae_local_num_queries, ae_global_dim_embed)
  │  (source input steps are summed into a single latent state, model.py:689)
  │
  ├─ for each output step: ForecastingEngine advances latent state autoregressively
  │    ├─ predict_decoders: per-stream coord embedding + TargetPredictionEngine + EnsPredictionHead
  │    └─ predict_latent:   LatentState + SSL heads (iBOT/JEPA/DINO)
  ▼
ModelOutput: .physical[step][stream] (ens, coords, channels), .latent[step][name]
```

HEALPix: the sphere is discretized at `cf.healpix_level`; `num_healpix_cells = 12 * 4**level`
(model.py:317, encoder.py:45, model.py:94). Level 5 → 12,288 cells. Each cell carries
`cf.ae_local_num_queries` latent tokens of width `cf.ae_global_dim_embed`. Cell neighbourhoods
(1-ring, 8 neighbours + self) are precomputed in `ModelParams.hp_nbours` via `astropy_healpix`
(model.py:140-157) and used by the decoders.

---

## 2. File-by-file overview

- **model.py** (839 lines) — Top-level `Model` (nn.Module): owns encoder, forecast engine, per-stream
  target-coordinate embeddings, target prediction engines, prediction heads, and SSL latent heads.
  Also `ModelParams` (non-trainable positional encodings, RoPE coords, HEALPix neighbour table) and
  `ModelOutput` (container for physical + latent predictions per forecast step). See §3.

- **encoder.py** (355 lines) — `EncoderModule`: orchestrates the embedding→local→global encoder
  hierarchy, including chunked processing of cells (flash-attn workaround), masked-cell handling,
  and register/class token insertion. See §4.

- **engines.py** (1152 lines) — All the "engine" building blocks used by encoder/model: embedding,
  local/global assimilation, query aggregation, forecasting, per-stream decoders, prediction heads,
  SSL latent heads, `LatentState`. See §5.

- **attention.py** (701 lines) — Attention primitives (self/cross, varlen/dense/local-block-sparse,
  sliced-Q). All built on flash-attn or flex_attention. See §6.

- **blocks.py** (263 lines) — Composite transformer blocks used by the newer decoder variants:
  `SelfAttentionBlock` (blocks.py:22), `CrossAttentionBlock` (blocks.py:85) — both optionally with
  DiT-style AdaLayerNorm conditioning — and `OriginalPredictionBlock` (blocks.py:180), a
  cross-attn(+optional self-attn)+MLP block conditioned on target coordinates.

- **ema.py** (116 lines) — `EMAModel` (ema.py:14): exponential-moving-average copy of the model,
  used as the teacher in student-teacher SSL. See §7.

- **embeddings.py** (144 lines) — Per-stream source embedding networks: `StreamEmbedTransformer`
  (embeddings.py:21; small transformer over channels with `full`/`block` unembed modes) and
  `StreamEmbedLinear` (embeddings.py:132). Selected per stream via `streams.<name>.embed.net`
  (`"transformer"` | `"linear"`).

- **layers.py** (95 lines) — `MLP` (layers.py:31; pre-LN, optional residual, optional
  AdaLayerNorm conditioning via `dim_aux`) and `NamedLinear` (layers.py:17; linear with a `.name`
  attribute so freeze/regex utilities can target it).

- **model_interface.py** (281 lines) — Model construction, FSDP2 sharding, checkpoint loading.
  Entry points `get_model` (model_interface.py:264), `init_model_and_shard`
  (model_interface.py:44), `load_model` (model_interface.py:177). See §8.

- **norms.py** (173 lines) — `RMSNorm` (norms.py:17), `AdaLayerNorm` (norms.py:64; scale/shift from
  an aux embedding), `AdaLayerNormLayer` (norms.py:105; DiT-style zero-init wrapper with gating,
  used by blocks.py), `SwiGLU`, `SaturateEncodings` (norms.py:162; soft clamp for VAE latents).

- **parametrised_prob_dist.py** (132 lines) — `DiagonalGaussianDistribution` (VAE posterior; kl/nll/
  sample) and `LatentInterpolator` (parametrised_prob_dist.py:80): projects local latents to
  mean/logvar, samples, optionally interpolates with noise. Enabled when
  `cf.latent_noise_kl_weight > 0`; the returned posteriors surface in
  `ModelOutput.latent[0]["posteriors"]` for the KL loss term.

- **positional_encoding.py** (175 lines) — `positional_encoding_harmonic` (sinusoidal PE added
  in-place to token sequences) plus variants, and 2D RoPE for lat/lon coordinates:
  `rotary_embedding_2d` / `rotary_pos_emb_2d` (positional_encoding.py:134/171), used by attention
  heads when `cf.rope_2D` is set.

- **ssl_target_processing.py** (265 lines) — Teacher-output post-processing for SSL losses, adapted
  from DINOv2: `iBOTPatchTargetProcessing` (ssl_target_processing.py:28) and `DINOTargetProcessing`
  (ssl_target_processing.py:150), each supporting `softmax_center` (EMA-centering + sharpening,
  distributed-aware async center updates) or `sinkhorn_knopp`; `JEPATargetProcessing`
  (ssl_target_processing.py:260) is identity. Consumed by the loss side, see
  `src/weathergen/train/DOCS-Train.md`.

- **utils.py** (79 lines) — `get_num_parameters`, `freeze_weights`, `apply_fct_to_blocks` (regex
  match on module `.name` — this is why many modules set `self.name`), `ActivationFactory`
  (string → activation for `pred_head.final_activation`).

- **ema.py / ssl_target_processing.py** interplay: see §7.

---

## 3. model.py — `Model`, `ModelParams`, `ModelOutput`

- `ModelOutput` (model.py:50) — `physical[fstep][stream_name]` is a tuple (split per batch sample)
  of tensors `(ens, num_coords, channels)`; `latent[fstep]` holds `"posteriors"`, `"latent_state"`
  (a `LatentState`), and one entry per SSL head (`"iBOT"`, `"JEPA"`, `"DINO"`).

- `ModelParams` (model.py:85) — *non-trainable* buffers created outside the model so they survive
  FSDP: local per-cell-token PE `pe_embed`, global per-cell PE `pe_global` (also gives masked cells
  identity — see comment model.py:216), optional 2D RoPE coordinates (`rope_coords`,
  `rope_cell_coords`, when `cf.rope_2D`), HEALPix neighbour table `hp_nbours`, and `q_cells_lens`
  for varlen attention. Instantiated in `model_interface.init_model_and_shard` (model_interface.py:170)
  and passed as first argument to `Model.forward`.

- `Model` (model.py:265)
  - `create()` (model.py:372) builds all submodules (construction is separate from `__init__` so it
    can run under `torch.device("meta")` for FSDP). Key decisions:
    - `EncoderModule` always.
    - `ForecastingEngine` if `cf.fe_num_blocks > 0`, else `IdentityEngine` (model.py:381).
    - Per-stream decoder stack only if a `LossPhysical` loss is enabled in
      `training_config.losses`/`validation_config.losses` (model.py:401). Per stream (skipping
      forcing-only streams): coord embedding (`NamedLinear` or `MLP`, per
      `streams.<name>.embed_target_coords.net`), then decoder chosen by `cf.decoder_type`:
      `"Linear"` → `BilinearDecoder`; `"PerceiverIOCoordConditioning"` →
      `TargetPredictionEngineClassic`; anything else → `TargetPredictionEngine`; plus an
      `EnsPredictionHead`. Streams can share the spatial decoder via `pred_spatial_shared`
      (model.py:493).
    - SSL latent heads from the `LossLatentSSLStudentTeacher` loss config (model.py:545):
      per loss fct (`iBOT`/`JEPA`/`DINO`) a `LatentPredictionHead{MLP,Transformer,Identity}`
      selected by `loss_cfg["head"]`, plus `latent_pre_norm` (LayerNorm).
  - `forward(model_params, batch)` (model.py:672):
    1. `self.encoder(...)` → global latent tokens + VAE posteriors.
    2. Reshape to `(batch, num_source_steps, tokens, dim)` and **sum over input steps** (model.py:691).
    3. For each output step: `forecast_engine(tokens, step, rope_coords)`; optional pushforward
       trick (`training_config.forecast.pushforward`) advances intermediate steps without grad and
       skips decoding (model.py:694).
    4. `predict_decoders` (model.py:734): strips aux (register/class) tokens, gathers 1-ring
       neighbourhood latents per cell via `model_params.hp_nbours` (9 tokens-groups per cell),
       embeds target coords, runs the per-stream decoder with varlen lens, applies
       `EnsPredictionHead`, splits back per batch sample. NaN coord embeddings are skipped with a
       warning (model.py:794).
    5. `predict_latent` (model.py:711): builds `LatentState` (register/class/patch split by
       `num_register_tokens`/`num_class_tokens`, model.py:660) and runs every SSL head.

---

## 4. encoder.py — `EncoderModule` hierarchy

`EncoderModule` (encoder.py:30). Construction order (encoder.py:59-118) — these are the verified
class names and sequence:

1. **`EmbeddingEngine`** (engines.py:36) — per-stream embedding.
2. **`LocalAssimilationEngine`** (engines.py:203) — self-attention within cells.
3. optional **`LatentInterpolator`** (parametrised_prob_dist.py:80) — if `cf.latent_noise_kl_weight > 0`.
4. **local→global adapter** — `Local2GlobalAssimilationEngine` (engines.py:247, cross-attention)
   or `Local2GlobalSumEngine` (engines.py:319, per-cell sum + projection) selected by
   `cf.ae_adapter_type` (`"cross_attention"` default | `"sum"`).
5. **`q_cells`** — learnable queries (encoder.py:88): shape
   `(num_healpix_cells, ae_local_num_queries, ae_global_dim_embed)` if
   `cf.ae_local_queries_per_cell` (with hand-coded cell metadata in the last ~10 channels),
   otherwise a single shared `(1, ae_local_num_queries, ae_global_dim_embed)` query.
6. **`QueryAggregationEngine`** (engines.py:376) — self-attention over per-cell queries of
   *unmasked* cells only.
7. **`GlobalAssimilationEngine`** (engines.py:455) — self-attention over all cells.

`forward` (encoder.py:120): `embed_engine` → `assimilate_local` → `ae_global_engine`; everything is
gradient-checkpointed.

`assimilate_local` (encoder.py:275) details worth knowing before editing:
- `cell_lens = batch.tokens_lens.sum(streams)` gives tokens-per-cell; cells with 0 tokens are
  "masked" (either no observations or SSL masking upstream in the dataset).
- Register + class tokens (`cf.num_register_tokens`, `cf.num_class_tokens`; default 0) are created
  from `q_cells` with harmonic PE and prepended per sample (encoder.py:295).
- `assimilate_local_project_chunked` (encoder.py:156) runs LocalAssimilationEngine + adapter in
  chunks over cells (flash-attn workaround; chunk size depends on `healpix_level`, encoder.py:167),
  applying the LatentInterpolator per chunk. Only unmasked cells go through the adapter.
- `aggregation_engine_unmasked` (encoder.py:218) packs unmasked-cell tokens + aux tokens per sample
  into a varlen sequence (with per-cell RoPE coords if enabled) for `QueryAggregationEngine`.
- Results are scattered back into the full grid (encoder.py:345); masked cells retain their
  initialization `q_cells + pe_global` — this is how masked-latent SSL targets get token identity.
- Output shape: `(batch*input_steps, (num_aux + num_cells) * ae_local_num_queries, ae_global_dim_embed)`.

Constraint asserts: `cf.ae_global_att_dense_rate == 1.0` and `forecast.att_dense_rate == 1.0` are
currently required when register tokens exist (encoder.py:65, model.py:333) — local block-sparse
attention doesn't handle the prepended aux tokens.

---

## 5. engines.py — engine and decoder classes

Encoder-side (wired in encoder.py, see §4):

- `EmbeddingEngine` (engines.py:36) — ModuleDict of per-stream `StreamEmbedTransformer` /
  `StreamEmbedLinear` / `Identity` (diagnostic or empty streams). Forward embeds each stream,
  then **reorders tokens from stream-major to cell-major** via `get_scatter_idxs_vectorized`
  (engines.py:175) and adds per-cell positional encoding `pe_embed` (asserts
  `ae_local_max_tokens_per_cell` isn't exceeded, engines.py:112).
- `LocalAssimilationEngine` (engines.py:203) — `ae_local_num_blocks` ×
  (`MultiSelfAttentionHeadVarlen` + `MLP`) in `ae_local_dim_embed`; the varlen lens are
  tokens-per-cell, so attention never crosses cell boundaries.
- `Local2GlobalAssimilationEngine` (engines.py:247) — `MultiCrossAttentionHeadVarlenSlicedQ`
  (queries = `q_cells` slices, KV = local cell tokens; maps `ae_local_dim_embed` →
  `ae_global_dim_embed`), then `(ae_adapter_num_blocks - 1)` × (MLP + cross-attn) more.
- `Local2GlobalSumEngine` (engines.py:319) — drop-in alternative (`ae_adapter_type: sum`):
  scatter-sum tokens per cell, linear projection, MLPs; same forward signature.
- `QueryAggregationEngine` (engines.py:376) — `ae_aggregation_num_blocks` ×
  (`MultiSelfAttentionHeadVarlen` + `MLP`); the local-attention branch controlled by
  `ae_aggregation_att_dense_rate` is currently `assert False` (incompatible with batching,
  engines.py:416), so keep the dense rate at 1.0.
- `GlobalAssimilationEngine` (engines.py:455) — `ae_global_num_blocks` blocks alternating dense
  `MultiSelfAttentionHead` and block-local `MultiSelfAttentionHeadLocal` per
  `ae_global_att_dense_rate` (last block always dense), each followed by `MLP`; optional trailing
  LayerNorm (`ae_global_trailing_layer_norm`).

Temporal:

- `ForecastingEngine` (engines.py:543) — the temporal autoregressive core. `fe_num_blocks` blocks
  alternating dense/local self-attention per `forecast_att_dense_rate`, each + MLP, optional
  LayerNorms after configured blocks (`fe_layer_norm_after_blocks`). All weights initialized near
  zero (engines.py:614) so an untrained engine is ≈ identity. Note: blocks are only built when
  `training_config.forecast.policy` is set (engines.py:559). During training it can add latent
  noise (`fe_impute_latent_noise_std`). Called once per output step in `Model.forward`.
- `IdentityEngine` (engines.py:532) — no-op stand-in when `fe_num_blocks == 0` (pure
  assimilation/SSL training).

Decoder-side (per stream, wired in model.py `create`):

- `TargetPredictionEngineClassic` (engines.py:694) — used when
  `cf.decoder_type == "PerceiverIOCoordConditioning"` (the classic readout): stack of
  `MultiCrossAttentionHeadVarlen` (Q = embedded target coords, KV = 1-ring neighbourhood latents)
  + optional self-attn (`cf.pred_self_attention`) + MLP, all conditioned on raw coordinates via
  AdaLayerNorm (`dim_aux`).
- `TargetPredictionEngine` (engines.py:801) — newer variant built from blocks.py, decoder type per
  `cf.decoder_type`: `PerceiverIO`, `AdaLayerNormConditioning`, `CrossAttentionConditioning`,
  `CrossAttentionAdaNormConditioning`, `PerceiverIOCoordConditioning` (docstring engines.py:822).
  Caveat: some of these paths look stale — `model.py:464` passes `stream_config=` but the
  signature takes `stream_name=`, and `next(self.cf.streams.values())` (engines.py:867) is not
  valid on a dict view — so expect to fix call-site plumbing if you enable a non-default
  `decoder_type` other than `"Linear"`/`"PerceiverIOCoordConditioning"`.
- `BilinearDecoder` (engines.py:1136) + `EfficientBilinear` (engines.py:1118) — used when
  `cf.decoder_type == "Linear"`: a single bilinear form (coords × latent → channels) as a cheap
  linear-probe decoder; bypasses `EnsPredictionHead` (ensemble dim 1).
- `EnsPredictionHead` (engines.py:639) — `ens_size` independent MLP heads mapping decoder tokens to
  physical channels; output `(ens, coords, channels)`; optional `final_activation`.

SSL latent heads:

- `LatentState` (engines.py:985) — dataclass with `class_token`, `register_tokens`, `patch_tokens`
  (post `latent_pre_norm`) and `z_pre_norm`.
- `LatentPredictionHeadMLP` (engines.py:1090), `LatentPredictionHeadTransformer` (engines.py:997;
  bottleneck linear + self-attn blocks + out projection), `LatentPredictionHeadIdentity`
  (engines.py:1079). Which tokens they consume (class/patch) is fixed per loss in `Model.create`:
  iBOT = class+patch, JEPA = patch, DINO = class (model.py:556).

---

## 6. attention.py — attention variants

All classes share the pattern: pre-norm (LayerNorm/RMSNorm, or `AdaLayerNorm` when `dim_aux` is
given), separate Q/K/V projections, optional QK-norm (`with_qk_lnorm`, type via `qk_norm_type`),
attention in `attention_dtype` (default bf16), output projection, optional residual. Most assert
`with_flash=True` — only flash paths are maintained.

- `MultiSelfAttentionHeadVarlen` (attention.py:27) — variable-length self-attention via
  `flash_attn_varlen_func`; sequences are packed flat with a lens tensor (leading 0 sentinel,
  cumsum'd inside). This is the workhorse for ragged per-cell / per-sample token sets. Optional 2D
  RoPE (`with_2d_rope` + `coords`).
- `MultiSelfAttentionHeadVarlenFlex` (attention.py:129) — experimental flex_attention version with
  a compiled sparsity mask; not wired into any engine currently.
- `MultiSelfAttentionHeadLocal` (attention.py:210) — block-local self-attention using
  `torch.nn.attention.flex_attention` with a precomputed block mask (`q_idx // block_factor ==
  kv_idx // block_factor`); the "local attention" alternative in the global/forecast engines
  (`*_att_dense_rate < 1`), grouping `block_factor` consecutive HEALPix tokens.
- `MultiCrossAttentionHeadVarlen` (attention.py:305) — varlen cross-attention
  (`flash_attn_varlen_func`), Q and KV with independent lens; used by decoders (targets attend to
  neighbourhood latents).
- `MultiCrossAttentionHeadVarlenSlicedQ` (attention.py:410) — cross-attention with one Q-projection
  per query slice (`num_slices_q = ae_local_num_queries`); used by the local→global adapter so each
  of the per-cell learnable queries gets its own projection.
- `MultiSelfAttentionHead` (attention.py:527) — dense (non-varlen) self-attention via
  `flash_attn_func` on batched fixed-length sequences; used by the global assimilation and forecast
  engines and stream embedders. Optional 2D RoPE.
- `MultiCrossAttentionHead` (attention.py:620) — dense cross-attention via torch SDPA
  (FLASH_ATTENTION backend); currently not used by the engines.

Memory efficiency comes from (a) varlen packing (no padding), (b) sliced-Q (queries chunked over
`ae_local_num_queries`), (c) block-local attention instead of dense, (d) gradient checkpointing at
the engine level, and (e) the cell-chunking in `encoder.assimilate_local_project_chunked`.

---

## 7. EMA and SSL (ema.py, ssl_target_processing.py)

`EMAModel` (ema.py:14, adapted from NVlabs/edm2) wraps *two* `Model` instances: the live student
(`original_model`) and a frozen copy (`ema_model`, the teacher). `update(cur_step, batch_size)`
lerps teacher params toward the student with beta from `halflife_steps` + `rampup_ratio`
(ema.py:65); `q_cells` and identity modules are excluded (ema.py:98). Handles FSDP-sharded and
DDP-prefixed state dicts. `forward_eval` runs the teacher under `no_grad` in eval mode.

Student-teacher SSL flow: the teacher (an `EMAModel` created by the training-side
`EMATeacher` target calculator — see `src/weathergen/train/DOCS-Train.md`) encodes the *unmasked*
batch; its `LatentState`/head outputs are post-processed by `ssl_target_processing.py`
(iBOT: per-patch centering+sharpening or Sinkhorn-Knopp; DINO: same for class tokens; JEPA:
identity) to form targets. The student encodes the masked batch (masked HEALPix cells fall back to
`q_cells + pe_global`, §4) and its `latent_heads` outputs (`ModelOutput.latent[step]`) are compared
against those targets by `LossLatentSSLStudentTeacher`. Teacher architecture can differ from the
student via `model_param_overrides` (plumbed through `get_model`'s `overrides`, §8).

A separate optional EMA instance can exist for validation (`validation_config.validate_with_ema`);
both are created training-side, ema.py just provides the mechanism.

---

## 8. Model construction and where to add things (model_interface.py)

Construction path: `Trainer` → `init_model_and_shard(cf, dataset, ...)` (model_interface.py:44) →
`get_model` (model_interface.py:264) → `Model(cf_with_overrides, sources_size,
targets_num_channels, targets_coords_size).create()`. The three size lists come from the dataset
(`dataset.get_sources_size()` etc.), which is why model shapes depend on the stream configs.
`overrides` are merged onto `cf` with `merge_configs` — this is the hook the EMA teacher uses to
build an architecturally different teacher.

`init_model_and_shard` also:
- builds on `meta` device when DDP+FSDP, then shards: attention/MLP module classes listed in
  `modules_to_shard` (model_interface.py:92) are individually `fully_shard`-ed per engine;
  forecast blocks use `reshard_after_forward=False` (needed for the rollout/pushforward loop,
  model_interface.py:113); target token engines are kept in fp32 policy (model_interface.py:135).
- applies `cf.freeze_modules` (regex on module `.name`) via `apply_fct_to_blocks`.
- loads checkpoints (`load_model`, model_interface.py:177 — tolerates missing/new modules and
  initializes them, useful for fine-tuning with added components).
- creates and initializes `ModelParams` separately from the model (model_interface.py:170).

**Adding a new architectural component, checklist:**
1. Implement the module (attention primitive → attention.py; composite block → blocks.py; engine →
   engines.py). Give it a `.name` attribute if it should be freezable via `cf.freeze_modules`.
2. Wire it in `EncoderModule.__init__`/`forward` (encoder.py) or `Model.create`/`forward`
   (model.py), gated on a config option read from `cf` (use `cf.get("my_option", default)` for
   backward compatibility with old checkpoints/configs; add the option to
   `config/default_config.yml`). Follow existing selector patterns: `ae_adapter_type`,
   `decoder_type`, `fe_num_blocks > 0`.
3. Construction must work on the `meta` device: no tensor *values* computed in `__init__` that need
   real data; put non-trainable precomputed tensors in `ModelParams` and initialize them in
   `reset_parameters`. Make sure your module has working `reset_parameters` semantics (Model.
   reset_parameters only resets `nn.Linear`/`nn.LayerNorm`, model.py:583).
4. If the module contains attention/MLP submodules of the types in `modules_to_shard`, FSDP
   sharding picks them up only if you add a loop for your new engine in `init_model_and_shard` —
   check whether one of the existing loops already covers it.
5. If it changes the latent token layout (aux tokens, queries per cell), update
   `tokens_to_latent_state` (model.py:660), `predict_decoders`'s aux-token stripping
   (model.py:763), and `ModelParams.pe_global`/`rope_coords`.

Key config knobs → architecture (all read from the merged OmegaConf `cf`):
`healpix_level`; `ae_local_{dim_embed,num_blocks,num_heads,num_queries,queries_per_cell,max_tokens_per_cell}`;
`ae_adapter_{type,embed,num_heads,num_blocks,...}`; `ae_aggregation_{num_blocks,num_heads,att_dense_rate,...}`;
`ae_global_{dim_embed,num_blocks,num_heads,att_dense_rate,block_factor,trailing_layer_norm}`;
`fe_{num_blocks,num_heads,dropout_rate,layer_norm_after_blocks,impute_latent_noise_std}` + `forecast_att_dense_rate`;
`decoder_type`; `pred_self_attention`; `pred_mlp_adaln`; `num_register_tokens`; `num_class_tokens`;
`rope_2D`; `with_flash_attention`; `attention_dtype`; `norm_type`/`qk_norm_type`/`norm_eps`/`mlp_norm_eps`;
`latent_noise_{kl_weight,gamma,use_additive_noise,deterministic_latents}`; `freeze_modules`;
per-stream: `streams.<name>.{embed,embed_target_coords,target_readout,pred_head,pred_spatial_shared,token_size}`.
