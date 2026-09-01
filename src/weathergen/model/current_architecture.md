# Current Model Architecture

## Scope

This note describes the current model assembly around `model.py`, `engines.py`, and
`layers.py`, with `encoder.py` included only where it explains the runtime flow.

The current pipeline works, but new options are expensive to merge because the same option
must be threaded through many constructors by hand.

## Easy Summary

- `Model.create()` builds the encoder, optional forecasting engine, per-stream physical
  decoders, and latent SSL heads.
- `EncoderModule` runs the embedding and assimilation pipeline once to produce global latent
  tokens.
- `Model.forward()` rolls those tokens forward in time and produces physical and latent outputs
  at each requested forecast step.

## Runtime Flow

```text
Model.forward(model_params, batch):
    output = ModelOutput(batch.output_len)

    tokens, posteriors = encoder(model_params, batch)
    output.latent[0]["posteriors"] = posteriors

    tokens = reshape(tokens, batch, input_steps)
    tokens = sum_over_input_steps(tokens)

    for step in batch.output_idxs:
        if forecast_engine exists:
            tokens = forecast_engine(tokens, step, coords=rope_coords)

        output = predict_decoders(model_params, step, tokens, batch, output)
        output = predict_latent(model_params, step, tokens, batch, output)

    return output
```

## Construction Flow

```text
Model.create():
    encoder = EncoderModule(cf, sources_size, targets_num_channels, targets_coords_size)

    if cf.fe_num_blocks > 0:
        forecast_engine = ForecastingEngine(cf, training_config, num_healpix_cells)

    if LossPhysical is enabled:
        for each stream without pred_spatial_shared:
            build target coordinate embedder
            build target token engine or linear decoder
            build ensemble prediction head

        for each stream with pred_spatial_shared:
            alias coordinate embedder and target token engine
            build its own ensemble prediction head

    latent_heads = {}
    for each enabled SSL loss bundle:
        if loss is iBOT:
            build head on class + patch tokens
        elif loss is JEPA:
            build head on patch tokens
        elif loss is DINO:
            build head on class tokens
```

## Encoder Flow

```text
EncoderModule.forward(model_params, batch):
    stream_cell_tokens = embed_engine(batch, pe_embed)

    tokens_global, posteriors = assimilate_local(model_params, stream_cell_tokens, batch)

    tokens_global = ae_global_engine(tokens_global, coords=rope_coords)

    return tokens_global, posteriors
```

```text
EncoderModule.assimilate_local(...):
    create query tokens per HEALPix cell
    run local assimilation on packed local tokens
    run local-to-global adapter on unmasked cells
    aggregate unmasked cell queries across the batch
    fill final global token tensor
```

## Main Components

| Component | Responsibility | Current construction style |
| --- | --- | --- |
| `EmbeddingEngine` | Stream-specific token embedding | Per-stream `if/elif` on `embed.net` |
| `LocalAssimilationEngine` | Local self-attention over packed cell tokens | Repeated `[self-attn, MLP]` blocks |
| `Local2GlobalAssimilationEngine` | Project local tokens into global query tokens | Manual cross-attention and MLP interleave |
| `QueryAggregationEngine` | Combine unmasked cell queries across the batch | Alternate local/global attention by index, then MLP |
| `GlobalAssimilationEngine` | Global latent mixing | Alternate local/global attention by index, then MLP |
| `ForecastingEngine` | Roll latent tokens forward in time | Same pattern again, plus optional trailing norms |
| `TargetPredictionEngineClassic` | Read out target tokens from latent state | Manual `[cross-attn, optional self-attn, MLP]` |
| `TargetPredictionEngine` | Decoder variants selected by `decoder_type` | Large `if/elif` recipe selection |
| `LatentPredictionHeadTransformer` | Transformer-style SSL head | Repeated `[self-attn, MLP]` blocks |
| `MLP` | Shared feedforward primitive | One implementation used everywhere |

## Pseudo-code By File

### `engines.py`

```text
for each engine class:
    read options directly from cf
    create attention module with long keyword list
    create MLP with another long keyword list
    repeat this pattern for every block in the stack
```

### `model.py`

```text
create encoder
create forecast engine

if physical loss enabled:
    create stream readouts in one loop
    patch shared readouts in a second loop

create latent heads by branching on loss name and head type
```

### `layers.py`

```text
MLP.__init__:
    choose norm
    append norm if requested
    append linear, activation, dropout repeatedly
    append output linear

MLP.forward:
    run layer list in order
    optionally add residual
```

## Problem Statement

The main problem is not just file length. The problem is that architectural choices are encoded as
many scattered constructor calls instead of a small number of reusable build rules.

Because of that, a new option such as `mlp_type`, `use_xsa`, `qk_norm_type`, or a deep SSL tap
point must be added in many places across `engines.py` and `model.py`.

## Duplication Hotspots

1. Repeated attention keyword plumbing in multiple engine classes.
2. Repeated MLP keyword plumbing in the same engine classes.
3. Repeated stack shapes: `[attention, MLP]`, `[cross-attn, optional self-attn, MLP]`, and
   alternating local/global attention schedules.
4. Decoder construction logic split across two target prediction engine implementations.
5. `Model.create()` mixes stream decoder assembly, shared-head aliasing, and latent-head routing in
   one long method.
6. Latent head routing is hard-coded by loss name instead of being described declaratively.

## Why This Hurts Branch Merges

- A feature branch that changes one architectural option often has to edit many call sites.
- Two branches can be logically compatible while still conflicting in the same constructor blocks.
- It is hard to review whether a feature changed behavior intentionally or only missed one of the
  duplicated call sites.