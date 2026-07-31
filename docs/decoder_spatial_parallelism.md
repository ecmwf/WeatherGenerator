# Decoder spatial parallelism

This document describes the HEALPix cell-parallel decoder introduced by commits:

| Commit | Purpose |
| --- | --- |
| `d753973b` | Parallelize decoder work by HEALPix cell |
| `6f087772` | Fix the Linear path, synchronize NaN handling, and remove a redundant lens collective |

The decoder feature extends the topology and contiguous cell ownership described in
[`encoder_spatial_parallelism.md`](encoder_spatial_parallelism.md). It deliberately reuses
`encoder_spatial_parallel_size`; there is no separate decoder-parallel configuration.

## Motivation

Before this change, every rank in an encoder spatial group decoded every target coordinate.
The ranks consumed the same sample and held the same global latent tensor, so this duplicated
coordinate embedding, prediction-engine, and prediction-head work across the group.

Decoder spatial parallelism distributes that work by output HEALPix cell:

1. every spatial rank retains the encoder's contiguous cell range;
2. target coordinates are restricted to cells in that range;
3. each owned cell receives its one-ring HEALPix neighborhood: itself and eight neighbors;
4. the rank decodes only its local target coordinates;
5. variable-length local predictions are gathered differentiably;
6. gathered rank shards are restored to sample-major, global cell order.

The complete post-forecast latent tensor is still replicated on each spatial rank. The feature
reduces decoder computation and intermediate activations; it does not shard the forecast engine
or its output tensor.

## Shared topology and configuration

The decoder copies these values from `EncoderModule` during model creation:

```text
decoder_spatial_parallel_group     = encoder.spatial_parallel_group
decoder_spatial_parallel_rank      = encoder.spatial_parallel_rank
decoder_spatial_parallel_size      = encoder.spatial_parallel_size
decoder_local_num_healpix_cells    = encoder.local_num_healpix_cells
decoder_local_cell_start           = encoder.local_cell_start
decoder_local_cell_end             = encoder.local_cell_end
```

The relevant configuration remains:

```yaml
encoder_spatial_parallel_size: 4
```

For HEALPix level 5 and four spatial ranks:

| Spatial rank | Decoder-owned cells | Cells/rank |
| ---: | ---: | ---: |
| 0 | `[0, 3072)` | 3072 |
| 1 | `[3072, 6144)` | 3072 |
| 2 | `[6144, 9216)` | 3072 |
| 3 | `[9216, 12288)` | 3072 |

Ranks in one spatial group consume the same batch. Different spatial groups remain data-parallel
and must not exchange decoder results with one another. For that reason, decoder collectives
always use `decoder_spatial_parallel_group`, not the default world process group.

With spatial size one, the decoder retains the original single-rank behavior and performs no
prediction gather.

## End-to-end data flow

For each decoded forecast step and output stream:

```text
global latent after assimilation/forecasting
    [batch, global_cells, queries_per_cell, latent_dim]
    │
    ├─ select each owned cell's self-plus-eight-neighbor indices
    ▼
rank-local packed neighborhoods
    [batch × local_cells × 9 × queries_per_cell, latent_dim]
    │
    ├─ slice globally constructed target coordinates by owned cell range
    ▼
rank-local packed target coordinates and per-cell lengths
    │
    ▼
coordinate embedding → prediction engine → prediction head
    │
    ▼
rank-local variable-length predictions
    │
    ├─ pad to the largest rank-local prediction count
    ├─ autograd-aware all-gather within the spatial group
    ├─ remove padding using globally known per-cell target lengths
    ▼
sample-major predictions in global HEALPix cell order
```

This process is repeated independently for every stream and every step on which the model calls
`predict_decoders()`.

## Nine-cell HEALPix neighborhoods

`ModelParams.hp_nbours` has shape:

```text
[num_healpix_cells, 9]
```

For each cell, column zero is the cell itself and the remaining columns are its eight one-ring
neighbors. HEALPix cells with a missing neighbor use the center cell as the replacement, so every
row has exactly nine valid indices.

`select_healpix_neighborhood_shard()` validates the global lookup shape, restricts its rows to
the rank-owned interval, and indexes the globally available latent tensor. The packed result
preserves this logical order:

```text
sample
    → owned cell
        → 9 neighborhood cells
            → query token
```

The implementation currently gathers the global latent at the encoder boundary and executes the
forecast engine globally on every spatial rank. Consequently, neighborhood construction is a
local tensor selection; it does not require a halo exchange between ranks.

For `PerceiverIOCoordConditioning`, each local target cell attends to the corresponding packed
nine-cell neighborhood. The default configuration uses one latent query per HEALPix cell.

## Target-coordinate sharding

Targets remain globally constructed on every rank in a spatial group. For one stream and forecast
step, each sample provides:

```text
target_coords       = packed target coordinates in cell-major order
target_coords_lens  = number of target coordinates in every global cell
```

Target counts vary by sample, stream, and cell. Directly slicing the packed coordinate tensor by
cell index would therefore be incorrect. `select_packed_cell_shard()` expands the local cell mask
by the per-cell lengths and returns complete coordinate segments for:

```text
[decoder_local_cell_start, decoder_local_cell_end)
```

The local lengths retain `(sample, local_cell)` order and are passed to the variable-length
prediction engine.

## Decoder paths

### PerceiverIOCoordConditioning

The default and shipped configurations use `PerceiverIOCoordConditioning`. Its local prediction
engine receives:

- locally owned target-coordinate embeddings;
- the packed nine-cell latent neighborhoods;
- one target length per local cell;
- one latent-neighborhood length per local cell;
- the local raw target coordinates used for conditioning.

The stream-specific ensemble prediction head then maps prediction tokens to physical channels.

### Linear

`Linear` remains a documented decoder option. `BilinearDecoder` expects one latent row for every
cell length entry, so spatial execution passes only:

```text
tokens[:, local_cell_start:local_cell_end]
```

rather than the global latent tensor. The current Linear implementation requires:

```yaml
ae_local_num_queries: 1
```

and raises `NotImplementedError` for any other query count.

## Empty local target domains and FSDP

A stream can have global targets while one spatial rank owns no target coordinates. That rank
must not skip FSDP-wrapped decoder modules while its peers enter them, because mismatched module
execution can deadlock forward or backward collectives.

For an empty local target domain, the decoder therefore:

1. creates one zero-valued dummy coordinate with the stream's coordinate width;
2. assigns that coordinate to one local cell for the variable-length call;
3. executes coordinate embedding, the prediction engine, and the prediction head;
4. discards the dummy prediction with an empty slice;
5. participates in the prediction gather with a valid empty local shard.

This preserves module-call order without adding a physical prediction.

If a stream has no target coordinates anywhere in the global batch, every rank skips that stream
before entering the local decoder path.

## Group-consistent NaN handling

After target-coordinate embedding, every spatial rank computes a local NaN flag. The flags are
reduced with `MAX` over the decoder spatial group before any rank enters the prediction engine or
prediction head.

If any rank reports a NaN, every rank raises `FloatingPointError`. Continuing with an empty result
would be unsafe because:

- ranks would enter FSDP-wrapped modules in different orders;
- gathered prediction lengths would no longer match the global target lengths;
- a later `torch.split()` could fail or silently misalign outputs.

## Variable-length prediction gather

### Why padding is required

`torch.distributed.nn.functional.all_gather` requires the same tensor shape on every rank, while
the number of target coordinates per rank can differ. Each rank derives every shard's packed
prediction count from the globally available `target_coords_lens`, then pads its local prediction
tensor to the largest count.

No collective is required for target lengths: every rank already has the same global target lens,
and `split_cell_lens_by_shard()` reconstructs the contiguous per-rank slices locally.

The padded predictions are gathered with the autograd-aware collective:

```python
gathered_pred = all_gather(
    pred,
    group=decoder_spatial_parallel_group,
)
```

Using the functional distributed gather preserves gradients from the reconstructed global loss
to the decoder computation on the rank that owned each target coordinate.

### Why gathered tensors cannot simply be flattened

Each rank packs all local samples before the gather:

```text
rank 0: [sample 0 local cells, sample 1 local cells, ...]
rank 1: [sample 0 local cells, sample 1 local cells, ...]
```

The gather returns rank-major order. Flattening it would produce:

```text
rank 0 sample 0
rank 0 sample 1
rank 1 sample 0
rank 1 sample 1
```

The output and loss code expect sample-major order:

```text
sample 0 rank 0 cells
sample 0 rank 1 cells
sample 1 rank 0 cells
sample 1 rank 1 cells
```

Flattening would also retain rank padding. It is only sufficient in the special case of local
batch size one with equal prediction counts on every rank.

### Reassembly

`reassemble_packed_cell_shards()` uses each rank's per-cell target lengths to:

1. calculate the valid prediction segment for each `(rank, sample)` pair;
2. remove padding;
3. concatenate rank segments for one sample in spatial-rank order;
4. concatenate the reconstructed samples in batch order.

Because spatial ranks own consecutive cell intervals, spatial-rank order is also global HEALPix
cell order. The reconstructed tensor has shape:

```text
[ensemble_size, total_global_target_coordinates, output_channels]
```

It is finally split by the original per-sample coordinate counts and stored in `ModelOutput`.

## Autoregressive rollout behavior

The decoder is invoked inside the loop over `batch.get_output_idxs()`. With:

```yaml
forecast:
  num_steps: 2
  offset: 1
```

the output indices are `[1, 2]`. The forecast engine advances the latent state and the decoder
runs at both steps. Therefore the prediction gather also occurs once per decoded stream at each
of those two steps.

When pushforward training is enabled, intermediate steps advance the forecast state without
decoding. Only the final gradient-bearing step executes the decoder and its gather.

## What is local and what remains global

| Stage/data | Spatially sharded? |
| --- | --- |
| Target reads and target-coordinate construction | No |
| Post-encoder/global-assimilation latent | No |
| Forecast engine and forecast latent | No |
| Selection of target coordinates by output cell | Yes |
| Selection of nine-cell neighborhoods | Yes |
| Coordinate embedding | Yes |
| Prediction engine | Yes |
| Prediction head | Yes |
| Reconstructed physical prediction | No |
| Physical loss after reconstruction | No |

The principal benefit is reduced decoder computation and activation memory. The principal added
cost is one differentiable prediction gather per decoded stream and forecast step, plus a small
NaN-flag reduction.

## Ordering guarantees

The implementation relies on these invariants:

1. all ranks in a spatial group process the same batch and target tensors;
2. decoder ownership exactly matches encoder ownership;
3. each rank owns a consecutive range of global nested HEALPix cell IDs;
4. target coordinates are packed in `(sample, cell)` order;
5. local coordinate selection preserves increasing global cell order;
6. process-group rank order matches cell-range order;
7. per-rank lengths and predictions use identical packing order;
8. reassembly removes padding before restoring sample-major order.

Violating any of these assumptions can produce a correct gather shape with incorrectly aligned
predictions, so ordering tests are as important as shape tests.

## Runtime verification

For HEALPix level 5, spatial size 4, local batch size `B`, and one query per cell, verify:

```text
global latent cells:             12288
local decoder cells/rank:         3072
neighborhoods per local cell:         9
packed neighborhood rows/rank: B × 3072 × 9
```

For each stream and decoded step:

- the sum of local target counts across spatial ranks equals the global target count;
- each local coordinate belongs to the rank's cell interval;
- gathered output length equals the sum of the original per-sample coordinate lengths;
- predictions have the same ordering and values as spatial size one, within numerical tolerance;
- backward reaches decoder parameters and the owning rank's local prediction graph.

Peak-memory comparisons should record every spatial rank. Sparse target streams can produce
imbalanced decoder work and memory even though cell ownership is equal.

## Troubleshooting

### Hang in decoder forward or backward

Verify that:

- all group ranks process the same stream and forecast step;
- empty local target ranks execute the dummy path;
- NaN handling is reduced over the spatial group;
- decoder collectives use the encoder spatial process group;
- no rank independently skips a globally non-empty target stream.

### Prediction gather shape mismatch

Check:

- identical ensemble size and channel count on every rank;
- correct global `target_coords_lens` on every rank;
- local prediction length equals the sum of the rank's local lens slice;
- local padding uses the maximum valid prediction count across the group;
- every rank enters one gather per stream and decoded step.

### Output ordering mismatch for batch size greater than one

Do not flatten the rank-major gather result. Use `reassemble_packed_cell_shards()` to transpose
the logical rank/sample order and remove padding.

### Linear decoder failure

Verify that the Linear path receives only the local latent cell slice and that
`ae_local_num_queries` is one.

### No decoder speedup

Confirm that target coordinates exist across the global domain and inspect target counts per
rank. The forecast engine and global latent are still replicated, so workloads dominated by
forecasting or global assimilation will not scale with decoder spatial size.

## File-level implementation map

| File | Responsibility |
| --- | --- |
| `config/default_config.yml` | Shared `encoder_spatial_parallel_size` and decoder type |
| `src/weathergen/model/model.py` | Local neighborhood/target selection, decoder execution, NaN synchronization, and prediction gather |
| `src/weathergen/model/spatial_parallel.py` | Packed cell selection, neighborhood selection, lens splitting, and prediction reassembly |
| `src/weathergen/model/encoder.py` | Spatial group and contiguous ownership reused by the decoder |
| `src/weathergen/utils/distributed.py` | Spatial group validation and construction |
| `tests/test_encoder_spatial_parallel.py` | Neighborhood selection, lens slicing, and result-order tests |

## Review checklist

- [ ] The decoder reuses `encoder_spatial_parallel_size` and the encoder process group.
- [ ] Cell ownership is identical in the encoder and decoder.
- [ ] Every local cell receives exactly nine neighborhood indices.
- [ ] Target-coordinate selection respects variable per-cell lengths.
- [ ] Linear decoding uses only local latent cells and one query per cell.
- [ ] Empty local target ranks enter all FSDP-wrapped decoder modules.
- [ ] NaN detection fails collectively before prediction modules.
- [ ] Prediction gathering remains autograd-aware.
- [ ] Target lens are derived locally instead of gathered redundantly.
- [ ] Padding is removed before output reconstruction.
- [ ] Multi-sample batches are restored to sample-major order.
- [ ] Every decoded autoregressive step performs matching collectives on all group ranks.

