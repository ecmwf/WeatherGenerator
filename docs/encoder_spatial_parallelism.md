# Encoder spatial parallelism

This document describes the encoder spatial-parallel implementation introduced by
[pkuyyj/WeatherGenerator PR #1](https://github.com/pkuyyj/WeatherGenerator/pull/1).
It covers the two feature commits:

| Commit | Purpose |
| --- | --- |
| `15dcff5a` | Add HEALPix encoder spatial parallelism |
| `db4b477d` | Build encoder inputs on rank-local HEALPix domains |

Upstream evaluation changes and branch-synchronization-only changes in the PR history are
outside the scope of this document.

## Motivation

WeatherGenerator receives several heterogeneous input streams. A stream can be globally
regular, such as ERA5, or contain a variable number of observations, such as
`METOP_ABC_AVHRR_IASI`. Before this change, every distributed rank embedded every source
token and performed local assimilation for every HEALPix cell.

Embedding and local assimilation are spatially independent until the local cell
representations are projected into the global latent representation. The implementation
therefore distributes these stages over a HEALPix domain:

1. ranks in one spatial group consume the same training sample;
2. every rank owns a disjoint set of complete HEALPix cells;
3. each of the nine streams is filtered and tokenized independently for that local domain;
4. embedding, local assimilation, and local-to-global projection operate on local data;
5. fixed-size per-cell latent representations are gathered in global cell order;
6. query aggregation and global assimilation continue with the reconstructed global tensor.

This reduces source-token and local-encoder activation memory. It does not shard the complete
model or the complete training step.

## Terminology

| Term | Meaning |
| --- | --- |
| `world_size` | Total number of distributed ranks |
| `spatial_parallel_size` | Number of ranks cooperating on one encoder input |
| spatial group | Consecutive ranks that consume the same sample and partition its HEALPix domain |
| data-parallel rank | Index of a spatial group in the global job |
| spatial rank | Rank index within a spatial group |
| data HEALPix level | The configured `healpix_level`, normally level 5 |
| packed tokens | Variable-length tensor containing only existing stream tokens |
| dense cell tensor | Fixed-size tensor with one position for every owned HEALPix cell |

For global rank `r` and spatial-parallel size `S`:

```text
data_parallel_rank = r // S
spatial_rank       = r % S
```

Ranks `[gS, gS + 1, ..., gS + S - 1]` form spatial group `g`.

For example, with eight total ranks and `S = 4`:

```text
spatial group 0: global ranks [0, 1, 2, 3]
spatial group 1: global ranks [4, 5, 6, 7]
```

The four ranks in each group consume the same sample. The two groups remain data-parallel
with respect to each other.

## HEALPix partitioning

### Number of cells

At HEALPix level `L`, the number of cells is:

```text
N(L) = 12 × 4^L
```

At the default data level `L = 5`:

```text
N(5) = 12 × 4^5 = 12,288
```

### Contiguous equal ownership

The configured data-level cells are divided directly into equal, consecutive rank-local
intervals:

```text
cells_per_rank = N(L) / S

cell_start = spatial_rank × cells_per_rank
cell_end   = cell_start + cells_per_rank
```

`N(L)` must be divisible by the spatial-parallel size. For level 5 and four spatial ranks:

| Spatial rank | Level-5 cells | Number of level-5 cells |
| ---: | ---: | ---: |
| 0 | `[0, 3072)` | 3072 |
| 1 | `[3072, 6144)` | 3072 |
| 2 | `[6144, 9216)` | 3072 |
| 3 | `[9216, 12288)` | 3072 |

Common group sizes produce:

| Spatial ranks | Level-5 cells/rank |
| ---: | ---: |
| 4 | 3072 |
| 8 | 1536 |
| 16 | 768 |
| 32 | 384 |

### Why consecutive ranges are valid

The input mapping uses nested HEALPix ordering:

```python
ang2pix(2**healpix_level, theta, phi, nest=True)
```

Nested ordering gives every data-level cell a stable integer ID. Assigning consecutive ID
intervals means that concatenating rank-local tensors in spatial-rank order reconstructs the
original global cell order. No coarser parent-cell constraint is required.

## Distributed topology and process groups

`get_encoder_spatial_parallel_size()` validates that:

- the configured spatial size is at least one;
- it does not exceed `world_size`;
- `world_size` is divisible by the spatial size.

`get_encoder_spatial_parallel_group()` creates process groups from consecutive global ranks.
All global ranks create the groups in the same order, and each process caches the group that
contains it.

With spatial size one, no additional process group is created and the feature reduces to the
single-rank compatibility behavior.

## End-to-end data flow

The source path can be summarized as:

```text
source readers
    │
    ├─ same sample on every rank in a spatial group
    │
    ▼
map every source location to nested HEALPix cell ID
    │
    ├─ filter cell_start <= cell_id < cell_end independently for each stream
    │
    ▼
construct local stream tokens and local per-cell token counts
    │
    ▼
embed every stream independently on its owning rank
    │
    ▼
scatter stream-major embeddings into local cell-major packed order
    │
    ▼
local assimilation
    │
    ▼
local-to-global projection
    │
    ▼
restore fixed-size dense local-cell tensor
    │
    ▼
autograd-aware all-gather in spatial-rank order
    │
    ▼
global cell tensor [0, N(healpix_level))
    │
    ▼
query aggregation and global assimilation
```

### What is local and what remains global

| Stage/data | Spatially sharded? |
| --- | --- |
| Source storage reads | No; each spatial rank currently reads the same source sample |
| Source coordinate-to-HEALPix mapping | Computed independently on every spatial rank |
| Source filtering and token construction | Yes |
| Per-stream embedding | Yes |
| Local assimilation | Yes |
| Local-to-global projection | Yes |
| Dense per-cell latent after gather | No |
| Query aggregation | No |
| Global assimilation | No |
| Target construction, prediction, and loss | No |
| Model parameters/FSDP state | Controlled separately by FSDP |

The implementation reduces GPU source-token and activation memory, but does not yet perform
distributed source I/O. Raw source reads and the coordinate mapping are replicated inside a
spatial group.

## Rank-local stream construction

### Filtering mask

Every stream is treated independently. After converting its coordinates to HEALPix IDs, the
rank-local point mask is:

```python
local_domain_mask = (cell_ids >= cell_start) & (cell_ids < cell_end)
```

`np.flatnonzero()` returns the original indices of retained points. The retained points are
then grouped by local HEALPix cell. The output is a list of length
`cell_end - cell_start`; empty cells remain present as empty index arrays.

This happens for every input stream, including:

- `METOP_ABC_AVHRR_IASI`;
- `ERA5_in`;
- `ERA5`;
- `METEOSAT_SEVIRI_IR`;
- `GOES_ABI_IR`;
- `HIMAWARI_AHI_IR`;
- `GOES_ABI_VIS`;
- `HIMAWARI_AHI_VIS`;
- `SurfaceCombined`.

Each stream can have a different number of retained observations and tokens on a rank.

### Local tokenization

`TokenizerMasking` receives `source_cell_start` and `source_cell_end`. Source windows call
`tokenize_space()` or `tokenize_spacetime()` with this interval. Target windows continue to
use the global interval.

For each local source cell:

1. locations are ordered by colatitude (`theta`);
2. ordered locations are split into patches of the stream's configured `token_size`;
3. the final patch is padded when required;
4. masking selects retained patches without changing their relative cell order;
5. `source_tokens_lens` records the number of retained patches in every local cell.

`StreamData` is initialized with:

```text
source HEALPix cells = local_num_healpix_cells
target HEALPix cells = num_healpix_cells
```

Consequently, for level 5 and four-way spatial parallelism:

```python
batch.source_samples.tokens_lens.shape[-1] == 3072
batch.target_samples.tokens_lens.shape[-1] == 12288
```

### Variable-length METOP observations

METOP is a useful example because its number of observations varies by time and domain.

Before tokenization, METOP-A, METOP-B, and METOP-C reader rows are concatenated in configured
reader order. Within each reader, source rows retain their storage order unless
`shuffle_source` is enabled.

For multiple source windows, `StreamData` stores step 0 as the newest window, followed by older
windows. This step order is retained when the embedding engine concatenates the stream inputs.

Tokenization deliberately converts this raw row order into cell-major order:

```text
local HEALPix cell 0
    locations ordered north-to-south
    patch 0: up to 512 locations
    patch 1: up to 512 locations
    ...
local HEALPix cell 1
    ...
```

A tensor such as:

```text
source_tokens_cells[step].shape = (N_rank, 512, 30)
```

has:

- a variable `N_rank`, the number of retained METOP patch tokens on this rank;
- 512 padded locations per patch;
- 30 encoded features per location.

`N_rank` does not need to be equal across spatial ranks. The variable-length METOP tensor is
never directly passed to a fixed-shape all-gather.

The temporary raw-row indices used by tokenization are not retained after source construction.
The implementation preserves equivalence with the original cell-major model order, but does
not provide an inverse map from assimilated latents back to original METOP-A/B/C storage rows.

## Embedding order

The embedding engine processes streams in configuration order. For each stream, it concatenates
source tokens over input steps and batch samples and applies the stream-specific embedding
network.

At this point, embeddings are stream-major. `get_scatter_idxs_vectorized()` uses
`batch.tokens_lens` to scatter them into packed cell-major order. Within each
`(input_step, sample, cell)` position, streams retain configuration order and each stream
retains its per-cell patch order.

The positional-encoding index is derived from the same per-cell counts. Therefore the packed
embedding tensor, its cell lengths, and its positional encodings remain aligned.

If a rank owns no observations from any stream for a sample, the embedding engine returns an
empty tensor rather than failing. The rank must still participate in later synchronized model
and distributed operations.

## Local assimilation and empty domains

`cell_lens_local` describes the packed local token tensor in:

```text
(input_step, sample, local_cell)
```

order. Local assimilation uses these lengths to derive cumulative packed-token boundaries.
Cell boundaries are never inferred by evenly slicing the token dimension, because token counts
vary by stream, time, cell, and rank.

When spatial parallelism is enabled, one local rank shard is processed as one local-assimilation
chunk. This keeps the number and order of calls to FSDP-wrapped local modules consistent across
spatial ranks.

A local domain can contain no observations. Skipping FSDP modules on that rank would cause
ranks to enter FSDP collectives in different orders and can deadlock backward. For an empty
chunk, the implementation therefore:

1. creates a one-token zero-valued dummy input;
2. calls the local assimilation engine;
3. calls latent interpolation;
4. calls the local-to-global adapter;
5. multiplies the result by zero and attaches it to the real output.

The zero dependency has no numerical effect, but preserves the forward/backward graph and
ensures that FSDP hooks execute consistently on every spatial rank.

## Local-to-global boundary and all-gather

The gather occurs after:

```text
stream embedding
    → local assimilation
    → latent interpolation
    → local-to-global projection
```

It occurs before:

```text
query aggregation
    → global assimilation
```

This is the principal synchronization boundary of the feature.

### Why variable-length source tensors can be gathered safely

The variable-length source tensors are first projected into a fixed number of query latents per
cell. Before gathering, every rank restores a dense local tensor:

```text
[
    input_steps × samples,
    local_num_healpix_cells,
    local_queries_per_cell,
    global_embedding_dimension,
]
```

Non-empty cells receive the projected local result. Empty cells retain their initialized latent
slot. All ranks in a spatial group therefore have the same gather shape even when their raw
observation and patch-token counts differ substantially.

The implementation uses `torch.distributed.nn.functional.all_gather`, not the non-autograd
collective, so gradients propagate from global processing back into the owning rank's local
encoder path.

### Reconstruction order

The spatial process group is created from increasing, consecutive global ranks. The gather
returns tensors in process-group rank order:

```text
[spatial rank 0, spatial rank 1, ..., spatial rank S - 1]
```

Each rank's local tensor is already ordered by increasing cell ID within its consecutive range.
Concatenating gathered tensors along the cell dimension therefore reconstructs:

```text
global cell 0, global cell 1, ..., global cell N - 1
```

For four ranks at level 5:

```text
gather result =
    rank 0 cells [0, 3072)
    + rank 1 cells [3072, 6144)
    + rank 2 cells [6144, 9216)
    + rank 3 cells [9216, 12288)
```

The rank-local `tokens_lens` tensors are gathered and concatenated along their cell dimension in
the same rank order. The reconstructed global cell mask is therefore aligned with the dense
global latent tensor.

The implementation then packs non-empty global cells for query aggregation and restores the
global dense layout afterward.

## Ordering guarantees

The feature guarantees model-equivalent cell ordering:

1. nested HEALPix maps every source location to a deterministic global cell ID;
2. each global cell belongs to exactly one spatial rank;
3. rank-local cell lists are ordered by increasing global cell ID;
4. local tokenization matches the corresponding slice of global tokenization;
5. stream embeddings are deterministically scattered using `tokens_lens`;
6. dense local latent position `j` maps to global cell `cell_start + j`;
7. all-gather results are concatenated in spatial-rank order;
8. gathered lengths and gathered latents use the same order.

This does not mean that the original source storage-row order survives tokenization. The model's
canonical order is cell-major, followed by per-cell location/patch order. That was already the
order consumed by local assimilation before spatial parallelism.

The local/global construction equivalence test builds the global cell lists and the four local
cell lists independently, concatenates the local lists, and checks exact array equality for every
cell.

### Sort stability caveat

On NumPy 2.x, cell grouping explicitly requests a stable sort. On NumPy 1.x, the compatibility
fallback currently uses NumPy's default `argsort`. The subsequent PyTorch colatitude sort is
stable, so ordinary observations remain deterministic, but exact relative ordering of duplicate
cell IDs with identical colatitudes is not formally guaranteed on the NumPy 1.x fallback.

If exact duplicate-location ordering is required across NumPy versions, use:

```python
np.argsort(local_cell_ids, kind="stable")
```

and retain an explicit raw observation identifier for inverse mapping.

## Backward-compatible full-grid input

The encoder accepts either:

- a rank-local source batch with `local_num_healpix_cells`; or
- a legacy global source batch with `num_healpix_cells`.

For a local batch, the embedding result is already local.

For a global batch, `select_packed_cell_shard()` selects complete cells from the packed token
tensor using `cell_lens`. This fallback preserves compatibility, but the selection occurs after
embedding, so it does not provide the embedding-memory reduction of rank-local source
construction.

`select_packed_cell_shard()`:

1. reshapes flattened cell lengths into complete HEALPix grids;
2. marks `[cell_start, cell_end)` in every input-step/sample row;
3. expands the cell mask by each cell's variable token count;
4. selects the corresponding packed tokens;
5. returns local cell lengths in unchanged packed order.

It validates the cell range and verifies that the lengths describe the actual packed tensor.

## Data-parallel training semantics

Spatial ranks consume the same sample and must not be counted as independent data-parallel
replicas.

The effective data-parallel world size is:

```text
data_parallel_world_size = world_size / encoder_spatial_parallel_size
```

The effective batch size is:

```text
effective_batch_size =
    batch_size_per_spatial_group × data_parallel_world_size
```

The trainer uses this effective data-parallel size for:

- total batch-size reporting;
- learning-rate scaling;
- scheduler construction;
- mini-epoch and continuation accounting.

The sampler similarly maps global ranks in the same spatial group to one data-parallel rank, so
they receive the same workset and random seed. Loader-worker IDs and mini-epoch indices still
differentiate independent workers and epochs.

When present in a saved run, `encoder_spatial_parallel_size_original` records the historical
spatial size so effective-batch calculations remain consistent. If the key is absent, the
implementation defaults it to the current spatial size; continuation from a pre-feature run
therefore requires checking this value explicitly.

## Configuration

The relevant option is:

```yaml
encoder_spatial_parallel_size: 4
```

`encoder_spatial_parallel_size` controls the number of ranks per spatial group.

Ready-to-use overrides are provided:

```text
config/encoder_spatial_parallel_4.yml
config/encoder_spatial_parallel_8.yml
```

For a four-rank job in which all ranks cooperate on one encoder sample:

```yaml
encoder_spatial_parallel_size: 4
```

The expected derived values are:

```text
world_size: 4
data_parallel_world_size: 1
local level-5 cells/rank: 3072
```

For an eight-rank job with two four-rank spatial groups:

```text
world_size: 8
encoder_spatial_parallel_size: 4
data_parallel_world_size: 2
```

### Continuing an existing run

A continued run can inherit a saved configuration that predates this feature. Pass the spatial
configuration explicitly when the intended current run should use spatial parallelism.

Verify the effective, merged configuration rather than relying only on the repository default.
In a four-rank spatial-only run, `data_parallel_world_size` must be 1, not 4.

## Runtime verification

### Startup log

Every spatial rank logs its fine-cell range. For level 5 and four ranks, expect:

```text
Encoder spatial rank 0/4 constructs source HEALPix cells [0, 3072)
Encoder spatial rank 1/4 constructs source HEALPix cells [3072, 6144)
Encoder spatial rank 2/4 constructs source HEALPix cells [6144, 9216)
Encoder spatial rank 3/4 constructs source HEALPix cells [9216, 12288)
```

### Tensor-shape checks

For level 5 and four spatial ranks:

```python
assert batch.source_samples.tokens_lens.shape[-1] == 3072
assert batch.target_samples.tokens_lens.shape[-1] == 12288
```

For a globally regular stream such as ERA5, each rank should retain approximately one quarter of
the source tokens. Exact equality depends on grid conventions and masks.

For regional or observational streams such as METOP and geostationary satellite streams, token
counts can be strongly imbalanced. The important invariants are:

- every retained source token belongs to the rank's cell interval;
- no source token belongs to two ranks;
- the union of rank-local cell intervals covers the full grid.

### Memory checks

Use PyTorch allocated-memory statistics when measuring the feature:

```python
torch.cuda.reset_peak_memory_stats()

# Run one representative training iteration.

peak_allocated = torch.cuda.max_memory_allocated() / 2**30
peak_reserved = torch.cuda.max_memory_reserved() / 2**30
```

`nvidia-smi` primarily reflects CUDA memory reserved by PyTorch. Released tensor memory remains
in the caching allocator and may not visibly decrease even when live tensor memory does.

Compare:

- peak allocated memory;
- peak reserved memory;
- per-stage peaks;
- all spatial ranks, because token imbalance can make one domain more expensive than another.

## Tests

`tests/test_encoder_spatial_parallel.py` covers:

| Test area | Invariant |
| --- | --- |
| Local construction equivalence | Concatenated local cell lists equal global construction cell-by-cell |
| Invalid local range | Out-of-range cell intervals are rejected |
| Packed-token selection | Complete cells are selected across multiple input-step/sample rows |
| Coverage and gradients | Shards cover every packed token exactly once and preserve gradients |
| Invalid packed ranges | Invalid cell intervals are rejected |
| Distributed size validation | Spatial groups must divide the distributed world |

Recommended validation commands are:

```bash
pytest -q tests/test_encoder_spatial_parallel.py
./scripts/actions.sh lint
./scripts/actions.sh unit-test
```

A multi-rank integration run remains necessary to validate FSDP collective ordering, runtime
memory, and real-stream token balance.

## Expected memory behavior

Memory savings are not expected to be exactly `1 / spatial_parallel_size`.

Memory that should decrease includes:

- rank-local source token tensors transferred to the GPU;
- stream-embedding activations;
- local-assimilation activations;
- local-to-global projection activations.

Memory that remains replicated or becomes global includes:

- model parameters, gradients, and optimizer state according to the FSDP configuration;
- source-reader CPU data before local filtering;
- target tensors and target-side computation;
- the dense gathered global latent tensor;
- query aggregation;
- global assimilation;
- forecast engine and prediction heads;
- CUDA allocator cache.

The rank with the most observations can determine the job's usable batch size. Equal numbers of
HEALPix cells do not imply equal numbers of stream tokens.

## Troubleshooting

### Embedding memory does not decrease

Check:

```python
batch.source_samples.tokens_lens.shape[-1]
```

If it is 12,288 at level 5 in a four-way run, the data pipeline constructed a global source
batch. The encoder can still select a local shard after embedding, but embedding remains global.

Verify:

- `encoder_spatial_parallel_size: 4` is present in the effective run configuration;
- `data_parallel_world_size: 1` for a four-rank spatial-only run;
- all four ownership messages appear in the startup log;
- each source stream has a rank-local token count.

### `nvidia-smi` remains high

Compare `torch.cuda.max_memory_allocated()` with `torch.cuda.max_memory_reserved()`. A large
difference usually represents reusable cached allocator segments, not live model tensors.

### A rank has no observations

This is valid for regional streams and sparse samples. The embedding engine returns an empty
packed tensor, and the local assimilation path executes a zero-valued dummy dependency to keep
FSDP calls synchronized.

### Gather shape mismatch

The gather must receive:

```text
[steps × samples, local_cells, queries_per_cell, embedding_dimension]
```

with identical `local_cells`, query count, dtype, and embedding dimension on all ranks in the
spatial group. Variable source-token counts must be resolved before this boundary.

### Global ordering mismatch

Verify together:

- the identical contiguous local range computed by the sampler and encoder;
- nested HEALPix mapping (`nest=True`);
- local `tokens_lens` cell dimension;
- rank order inside the spatial process group;
- concatenation along the cell dimension;
- identical ordering for gathered lengths and gathered latents.

Do not compare assimilated output against raw source-reader row order. Compare it against the
non-parallel model's cell-major token and latent order.

### FSDP hang during backward

Confirm that:

- all ranks in a spatial group consume the same sample;
- all ranks enter the local FSDP modules in the same order;
- empty ranks execute the dummy dependency path;
- `world_size` is divisible by the spatial size;
- no rank independently skips an otherwise valid sample because its local source domain is empty.

## File-level implementation map

| File | Responsibility |
| --- | --- |
| `config/default_config.yml` | Default spatial size |
| `config/encoder_spatial_parallel_4.yml` | Four-rank override |
| `config/encoder_spatial_parallel_8.yml` | Eight-rank override |
| `src/weathergen/datasets/healpix_domain.py` | Rank-local point grouping |
| `src/weathergen/datasets/multi_stream_data_sampler.py` | Shared spatial-group samples, local ownership, and runtime logging |
| `src/weathergen/datasets/stream_data.py` | Separate source-local and target-global cell counts |
| `src/weathergen/datasets/tokenizer_masking.py` | Local source tokenization range |
| `src/weathergen/datasets/tokenizer_utils.py` | Nested HEALPix mapping and local token construction |
| `src/weathergen/model/engines.py` | Local per-stream embedding and empty-domain handling |
| `src/weathergen/model/spatial_parallel.py` | Legacy packed-token shard selection |
| `src/weathergen/model/encoder.py` | Local assimilation, local-to-global projection, and differentiable gather |
| `src/weathergen/train/trainer.py` | Effective data-parallel batch and scheduler semantics |
| `src/weathergen/utils/distributed.py` | Spatial group validation and construction |
| `tests/test_encoder_spatial_parallel.py` | Cell ownership, ordering, coverage, validation, and gradient tests |

## Commit-by-commit design history

### `15dcff5a`: Add HEALPix encoder spatial parallelism

This commit establishes the initial model-side and distributed design:

- adds `encoder_spatial_parallel_size`;
- creates four-rank and eight-rank configuration overrides;
- creates consecutive-rank spatial process groups;
- makes spatial-group ranks consume the same data sample;
- changes effective batch and scheduler scaling from global world size to data-parallel group count;
- introduces `select_packed_cell_shard()` for variable-length cell-packed tensors;
- selects local cells before local assimilation;
- builds local latent query seeds and local positional encodings;
- performs local assimilation and local-to-global projection on each rank;
- restores dense local cells and gathers them before global processing;
- uses an autograd-aware gather;
- adds synchronized handling for empty local domains;
- adds initial unit tests for packed-cell coverage, gradients, and distributed-size validation.

At this stage, embedding still runs on the global packed source tensor and the result is selected
after embedding.

### `db4b477d`: Build encoder inputs on local HEALPix domains

This commit moves the domain boundary into the data pipeline:

- introduces rank-local HEALPix point grouping;
- passes source cell ranges into source tokenization;
- constructs only rank-local source cells and source token tensors;
- keeps targets global;
- adds separate source and target HEALPix dimensions to `StreamData`;
- makes embedding operate on local source tensors;
- gathers local `tokens_lens` to reconstruct the global mask;
- accepts both local and legacy global source batches;
- handles ranks with no embedded streams;
- validates local/global batch cell dimensions;
- adds exact local-construction-versus-global-slice tests.

This is the commit that enables embedding activation memory to scale with the local observation
domain.

## Review checklist

Before merging or extending this feature, verify:

- [ ] The effective spatial size is explicit in the run configuration.
- [ ] `world_size % encoder_spatial_parallel_size == 0`.
- [ ] The data-level HEALPix cell count is divisible by the spatial size.
- [ ] Every spatial group consumes identical samples.
- [ ] Every source stream constructs only local cells.
- [ ] Source and target cell dimensions remain intentionally different.
- [ ] Variable-length stream tokens are resolved into fixed-size cell latents before gather.
- [ ] Local lengths and local latents are gathered in identical rank order.
- [ ] The gather remains autograd-aware.
- [ ] Empty local domains enter all FSDP modules in consistent order.
- [ ] Effective batch size counts spatial groups, not individual spatial ranks.
- [ ] Tests cover exact ordering, token coverage, and gradients.
- [ ] Runtime validation compares allocated memory across all spatial ranks.
