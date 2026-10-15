# Target Model Architecture

## Design Goal

Keep the runtime model pipeline roughly the same, but make the implementation feel more like normal
PyTorch code.

This version avoids a heavy resolver/spec framework. It prefers:

- small reusable `nn.Module` blocks
- a few helper functions that centralize repeated kwargs
- short builder methods on engines and model classes
- small registries only where the code already has real branching points

The aim is to reduce duplication without inventing a second architecture language on top of
PyTorch.

## High-Level Shape

```text
Config
  -> shared kwargs helpers
  -> reusable composite nn.Module blocks
  -> engine-specific builder methods
  -> small registries for decoder and latent head variants
  -> Model.create() orchestration
```

## Why This Style Is More PyTorch/Pythonic

The earlier spec-heavy direction is valid, but it introduces more indirection than this codebase
probably needs.

This alternative is more natural for this repository because:

1. behavior still lives in `nn.Module` classes
2. construction still happens with explicit Python loops
3. config plumbing is centralized with plain helper functions or dicts
4. dynamic choices use simple registries instead of generic spec interpreters
5. you can debug it in an IDE without mentally unpacking another abstraction layer

## Core Principles

1. Prefer modules over specs.
2. Prefer helper functions over generic factories.
3. Use registries only at real feature boundaries.
4. Keep `forward()` logic in modules, not in config-driven dispatch code.
5. Centralize repeated kwargs once, but keep local structure explicit.

## Main Building Blocks

### 1. Shared Kwargs Helpers

Instead of a large resolver layer, keep a few narrow helpers that return normal Python dicts.

```text
def common_attention_kwargs(cf):
    return {
        "with_flash": cf.with_flash_attention,
        "norm_type": cf.norm_type,
        "qk_norm_type": cf.qk_norm_type,
        "norm_eps": cf.norm_eps,
        "attention_dtype": get_dtype(cf.attention_dtype),
        "use_xsa": cf.get("use_xsa", False),
    }


def common_mlp_kwargs(cf):
    return {
        "norm_type": cf.norm_type,
        "norm_eps": cf.mlp_norm_eps,
        "mlp_type": cf.get("mlp_type", "mlp"),
    }
```

These are intentionally boring. They exist only to stop repeating the same keyword lists all over
`engines.py`.

### 2. Reusable Composite Blocks

The current code repeatedly builds shapes like:

- self-attention then MLP
- cross-attention then optional self-attention then MLP
- alternating local/global attention with the same trailing MLP

Those should become a few small reusable modules.

#### SelfAttentionMLPBlock

```text
class SelfAttentionMLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        attention_cls,
        attention_kwargs,
        mlp_kwargs,
        mlp_hidden_factor,
        mlp_dropout,
        with_2d_rope=False,
        dim_aux=None,
    ):
        super().__init__()
        self.attn = attention_cls(
            dim_embed=dim,
            dim_aux=dim_aux,
            with_2d_rope=with_2d_rope,
            **attention_kwargs,
        )
        self.mlp = MLP(
            dim,
            dim,
            hidden_factor=mlp_hidden_factor,
            dropout_rate=mlp_dropout,
            with_residual=True,
            dim_aux=dim_aux,
            **mlp_kwargs,
        )

    def forward(self, x, x_lens=None, coords=None, aux=None):
        x = self.attn(x, x_lens=x_lens, coords=coords, aux=aux)
        if aux is None:
            x = self.mlp(x)
        else:
            x = self.mlp(x, x_lens, aux)
        return x
```

#### CrossAttentionMLPBlock

```text
class CrossAttentionMLPBlock(nn.Module):
    def __init__(
        self,
        dim_q,
        dim_kv,
        cross_attn_cls,
        cross_attn_kwargs,
        mlp_kwargs,
        mlp_hidden_factor,
        mlp_dropout,
        with_self_attention=False,
        self_attention_cls=None,
        self_attention_kwargs=None,
        dim_aux=None,
    ):
        super().__init__()
        self.cross_attn = cross_attn_cls(
            dim_embed_q=dim_q,
            dim_embed_kv=dim_kv,
            dim_aux=dim_aux,
            **cross_attn_kwargs,
        )
        self.self_attn = None
        if with_self_attention:
            self.self_attn = self_attention_cls(
                dim_embed=dim_q,
                dim_aux=dim_aux,
                **self_attention_kwargs,
            )
        self.mlp = MLP(
            dim_q,
            dim_q,
            hidden_factor=mlp_hidden_factor,
            dropout_rate=mlp_dropout,
            with_residual=True,
            dim_aux=dim_aux,
            **mlp_kwargs,
        )

    def forward(self, x, x_kv, x_lens=None, x_kv_lens=None, aux=None, coords=None):
        x = self.cross_attn(x, x_kv, x_lens, x_kv_lens, aux)
        if self.self_attn is not None:
            x = self.self_attn(x, x_lens=x_lens, coords=coords, aux=aux)
        if aux is None:
            x = self.mlp(x)
        else:
            x = self.mlp(x, x_lens, aux)
        return x
```

These modules remove duplication at the right level: repeated architecture patterns, not just raw
kwargs.

### 3. Small Runtime Carriers

I would still keep a couple of small dataclasses where they simplify interfaces.

```text
EncoderOutput:
    tokens
    posteriors
    taps = optional list of intermediate states

StreamDecoderBundle:
    coord_embed
    target_engine
    pred_head
```

That is still very normal Python. It just avoids tuple-position coupling.

## Engine Construction Style

The engines should stay as named modules, but they should build themselves using small internal
helpers.

### Local Assimilation Example

```text
class LocalAssimilationEngine(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        attn_kwargs = common_attention_kwargs(cf) | {
            "num_heads": cf.ae_local_num_heads,
            "dropout_rate": cf.ae_local_dropout_rate,
            "with_qk_lnorm": cf.ae_local_with_qk_lnorm,
        }
        mlp_kwargs = common_mlp_kwargs(cf)

        self.blocks = nn.ModuleList(
            [
                SelfAttentionMLPBlock(
                    dim=cf.ae_local_dim_embed,
                    attention_cls=MultiSelfAttentionHeadVarlen,
                    attention_kwargs=attn_kwargs,
                    mlp_kwargs=mlp_kwargs,
                    mlp_hidden_factor=2,
                    mlp_dropout=cf.ae_local_dropout_rate,
                )
                for _ in range(cf.ae_local_num_blocks)
            ]
        )

    def forward(self, tokens_c, cell_lens_c, use_reentrant):
        for block in self.blocks:
            tokens_c = block(tokens_c, x_lens=cell_lens_c)
        return tokens_c
```

### Global Assimilation Example

The global engine still needs alternating dense and local attention, but that does not require a
spec language. A plain helper method is enough.

```text
class GlobalAssimilationEngine(nn.Module):
    def __init__(self, cf, num_healpix_cells):
        super().__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.blocks = nn.ModuleList([self._make_block(i) for i in range(cf.ae_global_num_blocks)])
        if cf.get("ae_global_trailing_layer_norm", False):
            self.blocks.append(nn.LayerNorm(cf.ae_global_dim_embed, elementwise_affine=False))

    def _make_block(self, index):
        if self._use_dense_attention(index):
            attention_cls = MultiSelfAttentionHead
            attention_kwargs = common_attention_kwargs(self.cf) | {
                "num_heads": self.cf.ae_global_num_heads,
                "dropout_rate": self.cf.ae_global_dropout_rate,
                "with_qk_lnorm": self.cf.ae_global_with_qk_lnorm,
            }
        else:
            attention_cls = MultiSelfAttentionHeadLocal
            attention_kwargs = common_attention_kwargs(self.cf) | {
                "num_heads": self.cf.ae_global_num_heads,
                "qkv_len": self.num_healpix_cells * self.cf.ae_local_num_queries,
                "block_factor": self.cf.ae_global_block_factor,
                "dropout_rate": self.cf.ae_global_dropout_rate,
                "with_qk_lnorm": self.cf.ae_global_with_qk_lnorm,
            }

        return SelfAttentionMLPBlock(
            dim=self.cf.ae_global_dim_embed,
            attention_cls=attention_cls,
            attention_kwargs=attention_kwargs,
            mlp_kwargs=common_mlp_kwargs(self.cf),
            mlp_hidden_factor=self.cf.ae_global_mlp_hidden_factor,
            mlp_dropout=self.cf.ae_global_dropout_rate,
            with_2d_rope=self.cf.get("rope_2D", False),
        )
```

That is explicit, short, and PyTorch-native.

### Forecasting Example

The forecasting engine can use the same block class as the global engine, with different kwargs.
That means the duplication disappears because the block is shared, not because the engine becomes
generic.

## Decoder Strategy

The decoder variants are a real branching point, so this is where a small registry is justified.

```text
DECODER_BUILDERS = {
    "Linear": build_linear_decoder,
    "PerceiverIO": build_perceiver_decoder,
    "AdaLayerNormConditioning": build_adanorm_decoder,
    "CrossAttentionConditioning": build_cross_attention_decoder,
    "CrossAttentionAdaNormConditioning": build_cross_attention_adanorm_decoder,
    "PerceiverIOCoordConditioning": build_original_prediction_decoder,
}
```

Then `Model.create()` becomes:

```text
builder = DECODER_BUILDERS[cf.decoder_type]
bundle = builder(cf, stream_cfg, targets_num_channels[i_stream], targets_coords_size[i_stream])
```

Each builder returns a `StreamDecoderBundle`.

This keeps decoder differences readable, but stops them from spilling across one very long method.

## Latent Head Strategy

This is another place where a small registry is better than a heavy abstraction.

### Head Builder Registry

```text
LATENT_HEAD_BUILDERS = {
    "mlp": build_latent_mlp_head,
    "transformer": build_latent_transformer_head,
    "identity": build_latent_identity_head,
}
```

### Token Selector Registry

```text
LATENT_TOKEN_SELECTORS = {
    "JEPA": select_patch_tokens,
    "DINO": select_class_tokens,
    "iBOT": select_class_and_patch_tokens,
}
```

Then the logic in `Model.create()` becomes straightforward:

```text
for loss_name, loss_conf in ssl_target_losses.loss_fcts.items():
    head_builder = LATENT_HEAD_BUILDERS[loss_conf["head"].lower()]
    token_selector = LATENT_TOKEN_SELECTORS[loss_name]
    self.latent_heads[loss_name] = head_builder(cf, loss_name, loss_conf, token_selector)
```

That keeps the branching local and explicit.

## `Model.create()` In The Target Style

The main cleanup in `model.py` is not a new abstraction layer. It is just splitting one large
method into a few normal helper methods.

```text
def create(self):
    self.encoder = self._build_encoder()
    self.forecast_engine = self._build_forecast_engine()
    self._build_stream_decoders()
    self._apply_shared_stream_decoders()
    self._build_latent_heads()
    return self
```

### Stream Decoder Helper

```text
def _build_stream_decoders(self):
    self.embed_target_coords = nn.ModuleDict()
    self.target_token_engines = nn.ModuleDict()
    self.pred_heads = nn.ModuleDict()

    for i_stream, stream_cfg in enumerate(self.cf.streams):
        if is_stream_forcing(stream_cfg):
            continue
        if stream_cfg.get("pred_spatial_shared") is not None:
            continue

        bundle = self._build_stream_decoder_bundle(i_stream, stream_cfg)
        self.embed_target_coords[bundle.name] = bundle.coord_embed
        self.target_token_engines[bundle.name] = bundle.target_engine
        self.pred_heads[bundle.name] = bundle.pred_head
```

This is ordinary Python and much easier to step through.

## End-To-End Pseudo-code

```text
common kwargs helpers
    -> define shared attention and MLP defaults once

reusable composite blocks
    -> encode the repeated architecture patterns once

engine classes
    -> build ModuleLists with short helper methods

decoder registries and latent head registries
    -> isolate the real branching points

Model.create()
    -> orchestrates encoder, forecast engine, stream decoders, and latent heads
```

## What I Would Avoid

I would avoid introducing a general-purpose `build_engine(spec, shared_options)` layer unless the
codebase really starts supporting many more interchangeable architectures than it does now.

That style is powerful, but here it risks becoming abstraction for abstraction's sake.

## What Stays Stable

- `Model`, `EncoderModule`, and the engine class names can stay intact.
- existing packed-tensor and checkpointed forward paths can stay intact.
- the runtime pipeline does not need to be redesigned.

## What Improves

1. new options like `mlp_type`, `use_xsa`, and `qk_norm_type` are added in one helper, not many
   call sites
2. repeated attention-plus-MLP patterns become real shared modules
3. `model.py` becomes normal orchestration code again
4. branch merges should conflict less because constructor plumbing is centralized and engine bodies
   get shorter