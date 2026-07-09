# datasets/ — data loading & tokenization

- `data_reader_base.py` — reader abstractions; `data_reader_anemoi.py` (gridded), `data_reader_obs.py` (observations). Extra source-specific readers live in `packages/readers_extra`, looked up by stream type via its registry.
- `multi_stream_data_sampler.py` — assembles training samples across streams; `batch.py` — batch/sample metadata; `stream_data.py`, `memory_pinning.py` — stream tensors + pinned-memory transfer.
- `tokenizer.py`, `tokenizer_masking.py`, `tokenizer_utils.py`, `masking.py` — turn reader output into model tokens, incl. masking for SSL.
- Inline tests: `utils_test.py`.
- `agent_docs/data-pipeline.md` — the full path from stream configs to ModelBatch. Read before changing readers, tokenization, or batch assembly; update after. Masking/SSL sample composition: `agent_docs/ssl-training.md`.
