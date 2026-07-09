# src/weathergen — core model + training code

- `model/` — architecture; `datasets/` — data loading/tokenization; `train/` — training loop; `utils/` — CLI, logging, helpers. Each has its own AGENTS.md with detail.
- Entry point: `run_train.py`.
- Config objects come from `weathergen.common.config` (in `packages/common`), not from here.
- Unit tests sit inline as `*_test.py` next to the code; run `uv run --extra cpu pytest src/`.
- `agent_docs/training-step.md` — the end-to-end training step across datasets/model/train. Read before changing the trainer, model forward, or losses; update after.
- `agent_docs/ssl-training.md` — the SSL/student-teacher delta (masking, teachers, latent losses). Read before touching SSL or masking code; update after.
