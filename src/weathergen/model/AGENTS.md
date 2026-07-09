# model/ — architecture

- `model.py` — `Model`, `ModelParams`, `ModelOutput`: top-level assembly. `model_interface.py` — `init_model_and_shard` (model init + FSDP sharding).
- `engines.py` — the processing stages: Embedding, LocalAssimilation, Local2Global*, GlobalAssimilation, Forecasting, TargetPrediction engines + latent prediction heads and `LatentState`.
- `encoder.py` — `EncoderModule`; building blocks in `blocks.py`, `attention.py` (varlen flash-attn heads), `layers.py`, `norms.py`, `embeddings.py`, `positional_encoding.py`.
- `ema.py` — EMA weights; `ssl_target_processing.py` — SSL/JEPA target processing; `parametrised_prob_dist.py` — parametrised output distributions.
