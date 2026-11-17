# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The reader of a run, by the run's ``type``."""

from weathergen.evaluate.io.csv_reader import CsvReader
from weathergen.evaluate.io.merge_reader import WeatherGenMergeReader
from weathergen.evaluate.io.wegen_reader import WeatherGenJsonReader, WeatherGenZarrReader


def get_reader(
    reader_type: str,
    run: dict,
    run_id: str,
    private_paths: dict[str, str],
    region: str | None = None,
    metric: dict[str, object] | None = None,
    rank_sample_map: dict[str, tuple[list[int], int]] | None = None,
):
    """The reader of *run*.

    Parameters
    ----------
    reader_type:
        The run's ``type``: ``zarr``, ``anemoi-target``, ``csv``, ``json``, ``merge`` or
        ``jsonmerge``.
    run, run_id, private_paths:
        The run's config, its id and the private paths of the platform.
    region, metric:
        For the JSON readers: what to load.
    rank_sample_map:
        For the zarr types: the map of the run's rank files,
        ``{str(path): (local_samples, offset)}`` as ``WeatherGenZarrReader.rank_sample_map``
        gives it.  A reader seeded with it does not open the rank files to index them: the
        tasks of a chunked run carry the map of the master's reader.
    """
    if reader_type in ("zarr", "anemoi-target"):
        reader = WeatherGenZarrReader(run, run_id, private_paths, rank_sample_map=rank_sample_map)
    elif reader_type == "csv":
        reader = CsvReader(run, run_id, private_paths)
    elif reader_type == "json":
        reader = WeatherGenJsonReader(run, run_id, private_paths, region, metric)
    elif reader_type == "merge":
        reader = WeatherGenMergeReader(run, run_id, private_paths)
    elif reader_type == "jsonmerge":
        reader = WeatherGenMergeReader(
            run, run_id, private_paths, region, metric, reader_type="json"
        )
    else:
        raise ValueError(f"Unknown reader type: {reader_type}")
    return reader
