# Variable-Specific Channel Masking — Implementation Notes

## Key Insights

### Data indexing with `tokenize_spacetime`
- `tokenize_spacetime` creates per-timestep `rdata_cur` subsets, generating indices **local** to each timestep (1-based with padding offset). These indices do NOT map directly to the full `rdata.data` array.
- The O96 reduced Gaussian grid has ~39,936 gridpoints, which is NOT a multiple of HEALPix cell count (e.g., 3,072 at level 4). Pre-computing a mask tensor by tiling cell-level masks is **incorrect** because gridpoints don't map 1:1 to cells.
- **Correct approach**: After spatial masking, use `masked_points_per_cell` (already computed by the token mask logic) to determine which HEALPix cell each surviving data point belongs to, then look up the per-channel mask for that cell.

### Channel masking architecture
- Channel masks are generated per-stream in `Masker.generate_channel_masks()`, stored as `dict[str, np.ndarray]` in `SampleMetaData.global_params["channel_masks"]`.
- Masks are applied **only to source/encoder input** (not targets) — targets should predict all channels.
- The mask dict is threaded: `_get_batch → _build_stream_data → _build_stream_data_input → get_source → tokenize_apply_mask_source`.
- Inside `tokenize_apply_mask_source`, `torch.repeat_interleave(arange(num_cells), masked_points_per_cell)` gives per-point cell IDs for efficient vectorised channel mask lookups.

### Config and stream structure
- Channel masking config lives in stream YAML under `channel_masking:` with `enabled: true` and `autocorr:` dict.
- The integration test streams are in `integration_tests/streams_channel_masking/` (with smaller token_size/embed dims for speed).
- `default_config.yml` merges in `model_input.forecasting` in addition to whatever the custom config specifies. Check merged config output carefully.

### Approach B: Per-Channel Source/Target Split with Loss Masking
- Each channel has its own spatial mask determining the source/target split. A cell can be "source" for one channel and "target" for another.
- **Source side**: In `tokenize_apply_mask_source`, per-channel masks zero out channels at source cells where that channel's mask says "masked". The zeroed channels still flow through the encoder but carry no information for that variable.
- **Target side**: In `tokenize_apply_mask_target`, a `channel_loss_mask` tensor of shape `[num_target_points, num_channels]` is computed as the COMPLEMENT of the source-side channel masks. `loss_mask[i, c] = 1.0` means channel `c` was hidden at the source cell corresponding to target point `i`, so the loss should count. `loss_mask[i, c] = 0.0` means the channel was visible → skip loss.
- The `channel_loss_mask` flows through: `tokenize_apply_mask_target → get_target_values → StreamData.set_target_data → target_and_aux_module_base → loss_module_physical → lp_loss`.
- In `lp_loss`, the mask is applied as `diff_p * channel_loss_mask` with normalisation by `channel_loss_mask.sum(0).clamp(min=1)` per channel.

### Training results (Approach B, run iy4m8chp)
- 59 steps over 5 mini-epochs, loss: 2.54 → 1.76 (train), 2.32 → 2.10 (validation)
- Clean convergence with no errors or NaNs

### Common pitfalls
- `stream_info` keys are stage-prefixed: `train_source_channels`, `val_source_channels` etc.
- `MaskData.get_channel_masks(idx)` returns `None` when no channel masking is configured.
- `ChannelMaskingConfig.from_config` must handle OmegaConf → dict conversion via `.to_container()`.
- Running training on login nodes with `| head -N` will kill the training process when the pipe closes (SIGPIPE). Use `nohup ... &` or `tee` instead.
- The `multi_stream_data_sampler` must thread `channel_masks_dict` to BOTH `_build_stream_data_input` (source) AND `_build_stream_data` (target/values), otherwise the target-side loss mask won't be computed.
