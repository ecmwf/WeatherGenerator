"""
Utilities related to reading and writing metrics.

We use our own simple json-based format to abstract away various backends
 (our own pipeline, mlflow, wandb, etc.).
"""

import io
import json
import logging
from pathlib import Path

import polars as pl

# Known columns that are not scalar metrics:
_known_cols = {"weathergen.timestamp": pl.Int64, "weathergen.time": pl.Int64, "stage": pl.String}

_logger = logging.getLogger(__name__)


def _read_metrics_source(f: str | Path) -> str | Path | io.BytesIO:
    """Return a readable NDJSON source, excluding a partial final write if necessary."""
    path = Path(f)
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return f

    complete_data, separator, final_record = data.rpartition(b"\n")
    if not separator:
        return f

    try:
        json.loads(final_record)
    except json.JSONDecodeError:
        _logger.warning("Ignoring incomplete final metrics record in %s.", path)
        return io.BytesIO(complete_data + b"\n")
    return f


def read_metrics_file(f: str | Path) -> pl.DataFrame:
    """
    Loads a file of metrics.

    The resulting dataframe has the following format:
    - all columns in known_cols (if they exist in the file) have the right type
    - all other columns are of type float64 (including NaN values)
    """

    # All values are scalar, except for known values
    # The following point needs to be taken into account:
    # 1. The schema is not known in advance
    # 2. NaN is encoded as string
    # 3. numbers are encoded as numbers
    # The file needs to be read 3 times:
    # 1. Get the name of all the columns
    # 2. Find all the NaN values
    # 3. Read the numbers
    # 4. Merge the two dataframes

    source = _read_metrics_source(f)

    def read_ndjson(**kwargs) -> pl.DataFrame:
        if isinstance(source, io.BytesIO):
            return pl.read_ndjson(io.BytesIO(source.getvalue()), **kwargs)
        return pl.read_ndjson(source, **kwargs)

    # Find the list of all columns (read everything)
    df0 = read_ndjson(infer_schema_length=None)
    # Read with the final schema:
    schema1 = dict([(n, _known_cols.get(n, pl.Float64)) for n in df0.columns])
    df1 = read_ndjson(schema=schema1)
    # Read again as strings to find the NaN values:
    schema2 = dict([(n, _known_cols.get(n, pl.String)) for n in df0.columns])
    metrics_cols = [n for n in df0.columns if n not in _known_cols]
    df2 = read_ndjson(schema=schema2).cast(dict([(n, pl.Float64) for n in metrics_cols]))

    # Merge the two dataframes:
    for n in metrics_cols:
        df1 = df1.with_columns(
            pl.when(pl.col(n).is_not_nan()).then(df1[n]).otherwise(df2[n]).alias(n)
        )
    return df1


def get_train_metrics_path(base_path: Path, run_id: str) -> Path:
    """
    Return the path to the training metrics.json for a particular run_id. This is required for
    backwards compatibility after changing the name of the `results/{RUN-ID}/metrics.json` file to
    `results/{RUN-ID}/{RUN-ID}_train_metrics.json` to disambiguate `metrics.json`.
    See https://github.com/ecmwf/WeatherGenerator/issues/590 for details.
    """
    if (base_path / run_id / "metrics.json").exists():
        return base_path / run_id / "metrics.json"
    else:
        return base_path / f"{run_id}_train_metrics.json"
