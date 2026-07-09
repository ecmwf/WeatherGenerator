# src/weathergen — core model + training code

- `model/` — architecture: encoder, engines, blocks, attention, embeddings.
- `datasets/` — data readers, tokenizer, batching, multi-stream sampler.
- `train/` — training loop, loss calculator, collapse monitor.
- `utils/` — CLI (`cli.py`), logging, distributed helpers.
- Entry point: `run_train.py`.
- Config objects come from `weathergen.common.config` (in `packages/common`), not from here.
- Unit tests sit inline as `*_test.py` next to the code; run `uv run --extra cpu pytest src/`.
