# weathergen-evaluate — fast evaluation of model outputs

Reads inference-stage output (native Zarr, no format conversion) and produces scores and plots. Full docs: `README.md` here; config reference: `docs/evaluate_config_reference.md`; configs: `config/evaluate/`.

- Entry: `run_evaluation.py`.
- `io/` — output readers (wegen, csv, merge); `scores/` — score computation + orchestration (incl. PSD); `utils/` — regions, derived channels, climatology, array/dict helpers.
