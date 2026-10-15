# Refactor Plan

## Goal

Move from the current hand-wired construction style to a spec-and-factory architecture that makes
options easier to pipeline and reduces duplication across `engines.py`, `model.py`, and
`layers.py`.

Each step below is intentionally testable on its own.

## Step 1: Lock The Current Behavior

### Objective

Create or keep focused characterization tests before moving construction logic.

### Work

- Keep `tests/test_layers.py` as the characterization suite for MLP variants.
- Keep `tests/test_attention_xsa.py` as the characterization suite for XSA behavior.
- Add focused `Model.create()` tests for:
  - latent head routing by loss name
  - stream decoder construction
  - `pred_spatial_shared` aliasing
  - engine block counts for one small config

### Why First

This refactor changes construction code more than forward math. Characterization tests are the
fastest way to detect accidental rewiring.

### Test Gate

- layer tests pass
- XSA tests pass
- latent head creation tests pass
- one small model-construction smoke test passes

## Step 2: Centralize Option Resolution

### Objective

Stop repeating long keyword lists directly at every attention and MLP call site.

### Work

- Add pure helper functions that resolve shared attention options from `cf`.
- Add pure helper functions that resolve MLP options from `cf` and local overrides.
- Add stream-level helpers for target readout config.
- Add latent-head spec resolution from training and validation loss config.

### Scope

This step should not change engine structure yet. It only changes where constructor arguments are
assembled.

### Test Gate

- helper unit tests verify resolved options
- engine instantiation tests still produce the same module types and counts

## Step 3: Make `MLP` An Explicit Backend Wrapper

### Objective

Turn `layers.MLP` into a stable wrapper over selectable feedforward backends such as standard MLP
and SwiGLU.

### Work

- Add `mlp_type` support in `layers.py`.
- Use the shared resolver from Step 2 to pass MLP options consistently.
- Keep the public `MLP(...)` call shape stable for existing callers.

### Why Here

SwiGLU is already the cleanest example of why the current plumbing is too wide.

### Test Gate

- `tests/test_layers.py` passes
- one residual-shape smoke test passes
- invalid `mlp_type` raises cleanly

## Step 4: Extract Repeated Engine Stage Builders

### Objective

Replace repeated inline block assembly with shared stage builders.

### Work

- Add builders for repeated `[attention, MLP]` stacks.
- Add builders for cross-attention plus optional self-attention plus MLP stacks.
- Add support for alternating local/global schedules through a small schedule spec.
- Refactor these engines to consume stage builders:
  - `LocalAssimilationEngine`
  - `Local2GlobalAssimilationEngine`
  - `QueryAggregationEngine`
  - `GlobalAssimilationEngine`
  - `ForecastingEngine`
  - `LatentPredictionHeadTransformer`

### Constraint

Do not change the external engine forward signatures in this step.

### Test Gate

- per-engine instantiation tests confirm block layout
- small forward-shape smoke tests pass for each engine
- checkpointed forward paths still run for a minimal config

## Step 5: Split `Model.create()` Into Assembly Units

### Objective

Separate stream decoder assembly, shared decoder aliasing, and latent-head assembly.

### Work

- Extract `build_stream_decoder_bundles(...)`.
- Extract `apply_shared_stream_aliases(...)`.
- Extract `build_latent_heads(...)`.
- Replace hard-coded loss-name branches with a small token-policy registry.

### Why This Matters

Right now `Model.create()` mixes three unrelated responsibilities, which is one reason deep SSL and
new decoder options both end up editing the same method.

### Test Gate

- existing latent-head tests still pass
- new `pred_spatial_shared` tests pass
- physical decoder creation tests pass for linear and transformer-like decoders

## Step 6: Normalize Decoder Recipes

### Objective

Make decoder variants recipe-driven instead of hand-written as large branching blocks.

### Work

- Introduce a `DecoderRecipe` or equivalent small spec.
- Share the block-construction logic between `TargetPredictionEngineClassic` and
  `TargetPredictionEngine`.
- Keep wrapper classes if needed for compatibility, but move the duplicated assembly into one place.

### Constraint

Do not force a single giant decoder class if that makes forward logic less readable. Share the
construction rules first.

### Test Gate

- construction tests for each supported `decoder_type`
- one forward-shape test per decoder recipe
- no change in output tensor shapes for existing configs

## Step 7: Introduce Named Intermediate Outputs

### Objective

Prepare the pipeline for deep SSL and similar features without ad hoc tuple expansion.

### Work

- Add an explicit `EncoderOutput` carrier with `tokens`, `posteriors`, and optional `taps`.
- Thread optional intermediate taps through the encoder and model forward path.
- Keep the no-deep-SSL path identical in behavior.

### Why Late

This is the first step that likely touches `encoder.py` in addition to the three main target files.
It should happen after the builder and assembly cleanups, when the shape of the abstraction is
already stable.

### Test Gate

- deep SSL disabled: forward output matches current behavior
- deep SSL enabled: intermediate taps are present and correctly shaped
- latent output storage remains stable for existing SSL heads

## Step 8: Migrate Feature Branches Onto The New Seams

### Objective

Move incoming branch features onto the centralized option and spec pipeline.

### Work

- pipe `mlp_type` through shared MLP options
- pipe `use_xsa` through shared attention options
- pipe `qk_norm_type` through shared attention options
- represent alternative local-to-global adapters as adapter specs
- represent deep SSL taps as stage tap specs

### Test Gate

- SwiGLU tests pass
- XSA tests pass
- latent head tests pass
- deep SSL tests pass

## Suggested Stop Points

If you want to keep reviews small, the clean stop points are:

1. after Step 2
2. after Step 4
3. after Step 6
4. after Step 8

Those are the points where the code should still feel coherent and shippable.

## Non-Goals

- Do not rewrite the math of attention or decoding unless a feature requires it.
- Do not introduce a deep class hierarchy just to remove a few repeated lines.
- Do not merge all decoder behaviors into one unreadable universal forward function.

## Success Criteria

The refactor is successful if:

1. adding a new model option requires changing one resolver or one spec, not many constructor
   blocks
2. duplicated stack assembly in `engines.py` is materially reduced
3. `model.py` reads as an assembly file instead of a second implementation site
4. deep SSL, SwiGLU, and XSA can land with smaller, more local diffs