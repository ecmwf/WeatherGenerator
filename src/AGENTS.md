# src/weathergen — core model + training code

- `model/` — architecture; `datasets/` — data loading/tokenization; `train/` — training loop; `utils/` — CLI, logging, helpers. Each has its own AGENTS.md with detail.
- Entry point: `run_train.py`.
- Config objects come from `weathergen.common.config` (in `packages/common`), not from here.
- Unit tests sit inline as `*_test.py` next to the code; run `uv run --extra cpu pytest src/`.
