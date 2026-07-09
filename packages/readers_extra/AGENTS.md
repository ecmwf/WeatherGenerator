# weathergen-readers-extra — source-specific data readers

- `registry.py` — `get_extra_reader(stream_type)` maps stream type → reader class via lazy imports (no import-time sanity check; bad readers fail at runtime).
- One `data_reader_*.py` per source: iconart, grep, iconesm, cams, mesh, anemoi_operan, fesom.
- To add a reader: implement it here (base classes in `src/weathergen/datasets/data_reader_base.py`) and add a case to the registry match.
