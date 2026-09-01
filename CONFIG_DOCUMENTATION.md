# WeatherGenerator Configuration Documentation

This document provides comprehensive documentation for all configuration parameters in the WeatherGenerator project. Configuration uses OmegaConf and follows a hierarchical merging pattern:

```
base_config.yml -> private_config -> stream_configs -> CLI overrides
```

## Table of Contents

- [Model Architecture Parameters](#model-architecture-parameters)
- [Training Configuration](#training-configuration)
- [Validation Configuration](#validation-configuration)
- [Data Loading Configuration](#data-loading-configuration)
- [Loss Configuration](#loss-configuration)
- [Masking Strategies](#masking-strategies)
- [Stream Configuration](#stream-configuration)
- [Experiment Tracking Tags](#experiment-tracking-tags)

---

## Model Architecture Parameters

### Embedding Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_orientation` | string | `"channels"` | Embedding orientation mode |
| `embed_unembed_mode` | string | `"block"` | Embedding/unembedding mode |
| `embed_dropout_rate` | float | `0.1` | Dropout rate for embeddings |

### Local Assimilation Engine (Self-attention within HEALPix cells)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ae_local_dim_embed` | int | `1024` | Embedding dimension for local attention |
| `ae_local_num_blocks` | int | `2` | Number of transformer blocks |
| `ae_local_num_heads` | int | `16` | Number of attention heads |
| `ae_local_dropout_rate` | float | `0.1` | Dropout rate |
| `ae_local_with_qk_lnorm` | bool | `True` | Apply LayerNorm to Q/K projections |
| `ae_local_num_queries` | int | `1` | Number of query tokens |
| `ae_local_queries_per_cell` | bool | `False` | Whether queries are per-cell |

### Local-to-Global Adapter Layer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ae_adapter_num_heads` | int | `16` | Number of attention heads |
| `ae_adapter_embed` | int | `128` | Embedding dimension for adapter |
| `ae_adapter_with_qk_lnorm` | bool | `True` | Apply LayerNorm to Q/K projections |
| `ae_adapter_with_residual` | bool | `True` | Use residual connections |
| `ae_adapter_dropout_rate` | float | `0.1` | Dropout rate |

### Global Assimilation Engine (Self-attention across HEALPix cells)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ae_global_dim_embed` | int | `2048` | Embedding dimension for global attention |
| `ae_global_num_blocks` | int | `8` | Number of transformer blocks |
| `ae_global_num_heads` | int | `32` | Number of attention heads |
| `ae_global_dropout_rate` | float | `0.1` | Dropout rate |
| `ae_global_with_qk_lnorm` | bool | `True` | Apply LayerNorm to Q/K projections |
| `ae_global_att_dense_rate` | float | `1.0` | Attention density rate (currently fixed to 1.0 due to Triton issues) |
| `ae_global_block_factor` | int | `64` | Block factor for attention |
| `ae_global_mlp_hidden_factor` | int | `2` | MLP hidden layer multiplier |
| `ae_global_trailing_layer_norm` | bool | `False` | Add trailing LayerNorm |

### Query Aggregation Engine

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ae_aggregation_num_blocks` | int | `2` | Number of transformer blocks |
| `ae_aggregation_num_heads` | int | `32` | Number of attention heads |
| `ae_aggregation_dropout_rate` | float | `0.1` | Dropout rate |
| `ae_aggregation_with_qk_lnorm` | bool | `True` | Apply LayerNorm to Q/K projections |
| `ae_aggregation_att_dense_rate` | float | `1.0` | Precentage of attention layers that are dense |
| `ae_aggregation_block_factor` | int | `64` | Block factor for attention |
| `ae_aggregation_mlp_hidden_factor` | int | `2` | MLP hidden layer multiplier |

### Decoder Configuration

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `decoder_type` | string | `"PerceiverIOCoordConditioning"` | See below | Decoder architecture type |
| `pred_adapter_kv` | bool | `False` | | Use adapter for key/value in predictor |
| `pred_self_attention` | bool | `True` | | Enable self-attention in predictor |
| `pred_dyadic_dims` | bool | `False` | | Use dyadic dimensions in predictor |
| `pred_mlp_adaln` | bool | `True` | | Use AdaLN in MLP |

**Available `decoder_type` options:**

| Decoder Type | Description |
|--------------|-------------|
| `PerceiverIOCoordConditioning` | Coordinate-based conditioning with modified Adaptive LayerNorm (default) |
| `PerceiverIO` | Simple cross-attention layer (Perceiver architecture) |
| `AdaLayerNormConditioning` | Conditions only via Adaptive LayerNorm |
| `CrossAttentionConditioning` | Cross-attention with MLP |
| `CrossAttentionAdaNormConditioning` | Cross-attention with both self-attention and Adaptive LayerNorm |
| `Linear` | Simple linear projection |

### Forecast Engine Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fe_num_blocks` | int | `6` | Number of transformer blocks in forecast engine |
| `fe_num_heads` | int | `16` | Number of attention heads |
| `fe_dropout_rate` | float | `0.1` | Dropout rate |
| `fe_with_qk_lnorm` | bool | `True` | Apply LayerNorm to Q/K projections |
| `fe_layer_norm_after_blocks` | list | `[]` | Block indices after which to add LayerNorm (0-indexed) |
| `fe_impute_latent_noise_std` | float | `0.0` | Standard deviation for latent noise imputation |
| `forecast_att_dense_rate` | float | `1.0` | Precentage of attention layers that are dense |

### Token Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_class_tokens` | int | `1` | Number of class tokens |
| `num_register_tokens` | int | `7` | Number of register tokens |
| `healpix_level` | int | `5` | HEALPix grid level (level 5 = 12,288 cells globally) |

### Latent Noise / VAE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_noise_kl_weight` | float | `0.0` | KL divergence loss weight |
| `latent_noise_gamma` | float | `2.0` | Gamma parameter for latent noise |
| `latent_noise_saturate_encodings` | int | `5` | Encoding saturation parameter |
| `latent_noise_use_additive_noise` | bool | `False` | Use additive noise in latent space |
| `latent_noise_deterministic_latents` | bool | `True` | Use deterministic latents |

### Compute and Precision Settings

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `with_mixed_precision` | bool | `True` | | Enable mixed precision training |
| `with_flash_attention` | bool | `True` | | Enable Flash Attention 2 |
| `compile_model` | bool | `False` | | Enable torch.compile |
| `with_fsdp` | bool | `True` | | Enable FSDP2 (Fully Sharded Data Parallel) |
| `attention_dtype` | string | `"bf16"` | `bf16`, `float32` | Data type for attention |
| `mixed_precision_dtype` | string | `"bf16"` | `bf16`, `float32` | Data type for mixed precision |
| `mlp_norm_eps` | float | `1e-5` | | Epsilon for MLP normalization |
| `norm_eps` | float | `1e-4` | | Epsilon for layer normalization |
| `norm_type` | string | `"LayerNorm"` | `LayerNorm`, `RMSNorm` | Type of normalization |
| `freeze_modules` | string | `""` | | Regex pattern to freeze modules (e.g., `".*ERA5"`) |

---

## Training Configuration

Located under `training_config:` in YAML files.

### Basic Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_mode` | list | `["masking"]` | Training modes to use |
| `num_mini_epochs` | int | `32` | Number of mini-epochs |
| `samples_per_mini_epoch` | int | `4096` | Samples per mini-epoch |
| `shuffle` | bool | `True` | Shuffle training data |
| `start_date` | datetime | `1979-01-01T00:00` | Training data start date |
| `end_date` | datetime | `2022-12-31T00:00` | Training data end date |
| `time_window_step` | timedelta | `06:00:00` | Time step between windows |
| `time_window_len` | timedelta | `06:00:00` | Length of each time window |
| `window_offset_prediction` | int | `1` | Steps offset for prediction target |

**Available `training_mode` options:**

| Mode | Description |
|------|-------------|
| `masking` | Masked prediction training |
| `student_teacher` | Self-supervised learning with student-teacher |
| `latent_loss` | Latent space loss training |

### Learning Rate Scheduling

Located under `training_config.learning_rate_scheduling:`:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `lr_start` | float | `1e-6` | | Initial learning rate |
| `lr_max` | float | `5e-5` | | Maximum learning rate |
| `lr_final_decay` | float | `1e-6` | | Learning rate for decay phase |
| `lr_final` | float | `0.0` | | Final learning rate |
| `num_steps_warmup` | int | `512` | | Number of warmup steps |
| `num_steps_cooldown` | int | `512` | | Number of cooldown steps |
| `policy_warmup` | string | `"cosine"` | `cosine`, `linear` | Warmup schedule policy |
| `policy_decay` | string | `"constant"` | `constant`, `cosine`, `linear` | Decay schedule policy |
| `policy_cooldown` | string | `"linear"` | `linear`, `cosine` | Cooldown schedule policy |
| `parallel_scaling_policy` | string | `"sqrt"` | `sqrt`, `linear`, `none` | LR scaling for distributed training |

### Optimizer Configuration

Located under `training_config.optimizer:`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grad_clip` | float | `1.0` | Gradient clipping norm |
| `weight_decay` | float | `0.1` | Weight decay coefficient |
| `log_grad_norms` | bool | `False` | Log gradient norms |

**AdamW parameters** (under `optimizer.adamw:`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta1` | float | `0.975` | AdamW beta1 |
| `beta2` | float | `0.9875` | AdamW beta2 |
| `eps` | float | `2e-08` | AdamW epsilon |

### Forecast Configuration

Located under `training_config.forecast:`:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `time_step` | timedelta | `06:00:00` | | Time interval (fixed) between consecutive forecast steps |
| `num_steps` | int | `2` | | Number of autoregressive forecast steps |
| `policy` | string | `"fixed"` | `fixed`, `null` | Forecast policy |

---

## Validation Configuration

Located under `validation_config:`. Inherits from `training_config` with overrides.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `samples_per_mini_epoch` | int | `256` | Validation samples per mini-epoch |
| `shuffle` | bool | `False` | Shuffle validation data |
| `start_date` | datetime | `2023-10-01T00:00` | Validation data start date |
| `end_date` | datetime | `2023-12-31T00:00` | Validation data end date |
| `write_num_samples` | int | `0` | Number of validation samples to write to disk |
| `output_streams` | list/null | `null` | Output streams to write (null = all) |
| `validate_before_training` | bool/int | `False` | Run validation before training (int = batch size) |

### Validation EMA Configuration

Located under `validation_config.validate_with_ema:`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `True` | Enable EMA for validation |
| `ema_ramp_up_ratio` | float | `0.09` | EMA ramp-up ratio |
| `ema_halflife_in_thousands` | float | `1e-3` | EMA half-life (in thousands of steps) |

---

## Data Loading Configuration

Located under `data_loading:`:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_workers` | int | No | Number of data loader workers (default: 12) |
| `rng_seed` | int | **Yes** (`???`) | Random seed for data loading |
| `repeat_data_in_mini_epoch` | bool | No | Repeat data within mini-epoch (default: False) |

---

## Loss Configuration

Losses are configured under `training_config.losses:` as a dictionary of named loss terms.

### LossPhysical

Used for direct physical variable prediction:

```yaml
losses:
  physical:
    type: LossPhysical
    weight: 1.0
    target_and_aux_calc: Physical
    loss_fcts:
      mse:
        weight: 1.0
        target_source_correspondence: {0: {0: "complement"}}
```

**Available loss functions for LossPhysical:**

| Function | Description |
|----------|-------------|
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `rss` | Residual Sum of Squares |
| `rmse` | Root Mean Squared Error |
| `kernel_crps` | Kernel Continuous Ranked Probability Score |
| `gaussian_crps` | Gaussian CRPS |
| `mse_ens` | Ensemble MSE |
| `stats` | Statistical loss |
| `stats_normalized` | Normalized statistical loss |
| `stats_normalized_erf` | Statistical loss with error function |
| `lp_norm_X` | Lp norm (replace X with integer, e.g., `lp_norm_1`) |

### LossLatentSSLStudentTeacher

Used for self-supervised learning with student-teacher approaches:

```yaml
losses:
  student-teacher:
    type: LossLatentSSLStudentTeacher
    enabled: True
    weight: 1.0
    target_and_aux_calc: EMATeacher
    loss_fcts:
      JEPA:
        weight: 5
        out_dim: 2048
        loss_extra_args: {}
        target_source_correspondence: {1: {1: "subset"}}
```

**Available SSL loss functions:**

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `JEPA` | Joint-Embedding Predictive Architecture | `out_dim` |
| `iBOT` | Image BERT with Object Tokens | `out_dim`, `center_momentum`, `teacher_temp`, `teacher_style`, `student_temp` |
| `DINO` | Distillation with No Labels | `out_dim`, `center_momentum`, `teacher_temp`, `teacher_style`, `student_temp` |

### Target and Auxiliary Calculator Options

| Calculator | Description |
|------------|-------------|
| `Physical` | Direct physical target extraction |
| `EMATeacher` | Exponential Moving Average teacher model for SSL |

**EMATeacher configuration:**

```yaml
target_and_aux_calc:
  EMATeacher:
    ema_ramp_up_ratio: 0.09
    ema_halflife_in_thousands: 1e-3
    model_param_overrides: {ae_global_num_blocks: 10}
```

### Target Source Correspondence

The `target_source_correspondence` parameter defines the relationship between source (input) and target data:

```yaml
target_source_correspondence: {target_idx: {source_idx: relationship}}
```

**Available relationship types:**

| Relationship | Description |
|--------------|-------------|
| `independent` | Source and target are independent |
| `complement` | Source is complement of target (`~target_mask`) |
| `identity` | Source is identical to target |
| `subset` | Source is subset of target (`mask & target_mask`) |
| `disjoint` | Source is disjoint from target (`mask & ~target_mask`) |

---

## Masking Strategies

Configured under `training_config.model_input:` and `training_config.target_input:`:

```yaml
model_input:
  strategy_name:
    masking_strategy: "random"
    enabled: True
    num_samples: 1
    num_steps_input: 1
    masking_strategy_config:
      rate: 0.4
      rate_sampling: False
      diffusion_rn: True
```

### Available Masking Strategies

| Strategy | Description |
|----------|-------------|
| `random` | Random token masking |
| `healpix` | HEALPix cell-based masking |
| `cropping_healpix` | Spatially contiguous HEALPix cropping |
| `forecast` | Forecasting mode (no masking) |
| `causal` | Causal masking (temporal) |

### Masking Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `rate` | float | Masking rate (0-1) |
| `rate_sampling` | bool | Sample rate from distribution |
| `hl_mask` | int | HEALPix level for masking |
| `diffusion_rn` | bool | Enable diffusion-based masking |
| `method` | string | Method for cropping_healpix: `disk`, `random_walk`, `geodesic_disk` |

---

## Stream Configuration

Streams are configured in YAML files under `config/streams/` directories and referenced via:

```yaml
streams_directory: "./config/streams/era5_1deg/"
streams: ???  # Loaded from streams_directory
```

### Stream Types

| Type | Description |
|------|-------------|
| `anemoi` | Gridded reanalysis data (e.g., ERA5) |
| `obs` | Observation data (surface, satellite, conventional) |
| `fesom` | Ocean/FESOM model data |
| `cams` | Atmospheric composition data |
| `iconart` | ICON model data with aerosol/tracer |
| `iconesm` | ICON ESM model data |

### Core Stream Parameters

```yaml
StreamName:
  type: anemoi
  filenames: ['data.zarr']
  stream_id: 0

  # Variable selection
  source_exclude: [list_of_variables]
  target_exclude: [list_of_variables]
  source: [explicit_variable_list]
  target: [explicit_variable_list]

  # Loss and weighting
  loss_weight: 1.0
  location_weight: null  # or "cosine_latitude"
  masking_rate: 0.6
  masking_rate_none: 0.05

  # Tokenization
  token_size: 8  # 8, 16, 32, 64, 128
  tokenize_spacetime: True
  max_num_targets: -1  # -1 = unlimited
  diagnostic: False
```

### Stream Embedding Configuration

```yaml
  embed:
    net: transformer  # transformer, linear
    num_tokens: 1
    num_heads: 8
    dim_embed: 256
    num_blocks: 2

  embed_target_coords:
    net: transformer
    dim_embed: 128
```

### Stream Target Readout Configuration

```yaml
  target_readout:
    type: obs_value  # obs_value, token
    num_layers: 2
    num_heads: 4
    sampling_rate: 0.2  # optional
```

### Stream Prediction Head Configuration

```yaml
  pred_head:
    ens_size: 1
    num_layers: 1
```

### Location Weight Options

| Option | Description |
|--------|-------------|
| `null` | No location weighting (default) |
| `cosine_latitude` | Weight by cosine of latitude (poles have less weight) |

---

## General Configuration

Located under `general:`:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `istep` | int | No | Current training step (mutable, default: 0) |
| `rank` | int | Set in program | Distributed training rank |
| `world_size` | int | Set in program | Total number of distributed workers |
| `multiprocessing_method` | string | No | Multiprocessing method (default: "fork") |
| `desc` | string | No | Run description |
| `run_id` | string | Can be generated | Unique run identifier |
| `run_history` | list | No | History of run IDs (for continuation) |

---

## Logging Configuration

Located under `train_log_freq:`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `terminal` | int | `10` | Terminal logging frequency (batches) |
| `metrics` | int | `20` | Metrics logging frequency (batches) |
| `checkpoint` | int | `250` | Checkpoint saving frequency (batches) |

---

## Experiment Tracking Tags

Located under `wgtags:`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `org` | string | Organization name (e.g., "ecmwf", "cmcc", "metnor") |
| `issue` | int | GitHub issue number for this experiment |
| `exp` | string | Experiment codename (e.g., "rollout_ablation_grid") |
| `grid` | string | Grid configuration tag |

---

## Configuration File Examples

### Pure Physical Training (default_config.yml style)

```yaml
training_config:
  training_mode: ["masking"]

  losses:
    physical:
      type: LossPhysical
      loss_fcts:
        mse: {}

  model_input:
    forecasting:
      masking_strategy: "forecast"

  forecast:
    time_step: 06:00:00
    num_steps: 2
    policy: "fixed"
```

### Physical + JEPA Training (config_physical_jepa.yml style)

```yaml
training_config:
  training_mode: ["masking", "student_teacher"]

  losses:
    physical:
      type: LossPhysical
      weight: 0.7
      loss_fcts:
        mse:
          weight: 0.8
          target_source_correspondence: {0: {0: "complement"}}
      target_and_aux_calc: Physical

    student-teacher:
      type: LossLatentSSLStudentTeacher
      weight: 1.0
      loss_fcts:
        JEPA:
          weight: 5
          out_dim: 2048
          target_source_correspondence: {1: {1: "subset"}}
      target_and_aux_calc: EMATeacher
```

### DINOv2 Training (config_dinov2.yml style)

```yaml
training_config:
  training_mode: ["masking", "student_teacher"]

  losses:
    student-teacher:
      type: LossLatentSSLStudentTeacher
      weight: 1.0
      loss_fcts:
        iBOT:
          weight: 0.75
          out_dim: 4096
          center_momentum: 0.9
          teacher_temp: 0.1
          teacher_style: "softmax_center"
          target_source_correspondence: {0: {0: "subset"}, 1: {3: "subset"}}
        DINO:
          weight: 0.25
          out_dim: 4096
          teacher_temp: 0.1
          teacher_style: "softmax_center"
          target_source_correspondence: {0: {1: "subset", 2: "identity"}}
      target_and_aux_calc: EMATeacher
```

---

## Quick Reference: All Enum/Option Values

| Parameter | Available Values |
|-----------|------------------|
| `stream.type` | `anemoi`, `obs`, `fesom`, `cams`, `iconart`, `iconesm` |
| `decoder_type` | `PerceiverIOCoordConditioning`, `PerceiverIO`, `AdaLayerNormConditioning`, `CrossAttentionConditioning`, `CrossAttentionAdaNormConditioning`, `Linear` |
| `training_mode` | `masking`, `student_teacher`, `latent_loss` |
| `loss_fcts` (Physical) | `mse`, `mae`, `rss`, `rmse`, `kernel_crps`, `gaussian_crps`, `mse_ens`, `stats`, `stats_normalized`, `stats_normalized_erf`, `lp_norm_X` |
| `loss_fcts` (SSL) | `JEPA`, `iBOT`, `DINO` |
| `target_and_aux_calc` | `Physical`, `EMATeacher` |
| `target_readout.type` | `obs_value`, `token` |
| `embed.net` | `transformer`, `linear` |
| `location_weight` | `null`, `cosine_latitude` |
| `masking_strategy` | `random`, `healpix`, `cropping_healpix`, `forecast`, `causal` |
| `target_relationship` | `independent`, `complement`, `identity`, `subset`, `disjoint` |
| `cropping_method` | `disk`, `random_walk`, `geodesic_disk` |
| `norm_type` | `LayerNorm`, `RMSNorm` |
| `mixed_precision_dtype` | `bf16`, `float32` |
| `attention_dtype` | `bf16`, `float32` |
| `policy_warmup` | `cosine`, `linear` |
| `policy_decay` | `constant`, `cosine`, `linear` |
| `policy_cooldown` | `linear`, `cosine` |
| `parallel_scaling_policy` | `sqrt`, `linear`, `none` |
| `forecast.policy` | `fixed`, `null` |
