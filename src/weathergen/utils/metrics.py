"""
Utilities related to reading and writing metrics.

We use our own simple json-based format to abstract away various backends (our own pipeline, mlflow, wandb, etc.).
"""

import polars as pl

# Known columns that are not scalar metrics:
_known_cols = {"weathergen.timestamp": pl.Int64, "weathergen.time": pl.Int64, "stage": pl.String}


def read_metrics_file(f: str) -> pl.DataFrame:
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


    # Find the list of all columns (read everything)
    df0 = pl.read_ndjson(f, infer_schema_length=None)
    # Read with the final schema:
    schema1 = dict([(n, _known_cols.get(n, pl.Float64)) for n in df0.columns])
    df1 = pl.read_ndjson(f, schema=schema1)
    # Read again as strings to find the NaN values:
    schema2 = dict([(n, _known_cols.get(n, pl.String)) for n in df0.columns])
    metrics_cols = [n for n in df0.columns if n not in _known_cols]
    df2 = pl.read_ndjson(f, schema=schema2).cast(dict([(n, pl.Float64) for n in metrics_cols]))

    # Merge the two dataframes:
    for n in metrics_cols:
        df1 = df1.with_columns(
            pl.when(pl.col(n).is_not_nan()).then(df1[n]).otherwise(df2[n]).alias(n)
        )
    return df1
