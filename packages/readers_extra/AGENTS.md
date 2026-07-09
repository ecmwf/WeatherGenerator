# weathergen-readers-extra — source-specific data readers

- `registry.py` — `get_extra_reader(stream_type)` maps stream type → reader class via lazy imports (no import-time sanity check; bad readers fail at runtime).
- One `data_reader_*.py` per source: iconart, grep, iconesm, cams, mesh, anemoi_operan, fesom.
- To add a reader: `agent_docs/recipes/add-data-reader.md`. Where readers sit in the pipeline: `agent_docs/data-pipeline.md`.
