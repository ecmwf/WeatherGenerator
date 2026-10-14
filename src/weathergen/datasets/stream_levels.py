# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Per-stream latent-level assignment for the multi-resolution latent.

This is the only place where a stream name meets a healpix level, so adding a stream
stays pure configuration and the pyramid / cascade code never mentions streams.
Two mappings are produced, both keyed by stream name:

  encode_level[stream]  -> the latent level the stream is tokenized at and written into
                           (and at which its target coordinates are tokenized).
  decode_levels[stream] -> the latent levels its decoder reads. By default a stream at
                           the coarsest level reads only its own level, and a stream at a
                           finer level reads every level from the coarsest up to its own,
                           so its prediction sees the large-scale context and local detail.

Per-stream config keys (all optional with a single-level ladder):

  latent_level:  explicit encode level (must be on the ladder);
  resolution_km / resolution_deg: nominal resolution, snapped onto the ladder: the stream
                 goes to the finest level whose cell size is still >= its resolution, so
                 it does not over-resolve the stream (>= ~1 point per cell);
  decode_levels: explicit list of decode levels (all <= the encode level, which must be
                 included).

With a single-level ladder every stream is assigned that level for encode and decode,
i.e. the single-latent behaviour.
"""

import logging

_logger = logging.getLogger(__name__)

# Approximate HEALPix cell size: 58.6 deg / 2**level, times 111.195 km/deg
_DEG_PER_CELL_CONST = 58.6
_KM_PER_DEG = 111.195


def cell_km(level: int) -> float:
    """Approximate healpix cell size in km at `level`."""
    return (_DEG_PER_CELL_CONST / (2**level)) * _KM_PER_DEG


def _resolution_km(stream_cfg) -> float | None:
    """Nominal stream resolution in km (`resolution_km` or `resolution_deg`), or None."""
    if "resolution_km" in stream_cfg:
        return float(stream_cfg["resolution_km"])
    if "resolution_deg" in stream_cfg:
        return float(stream_cfg["resolution_deg"]) * _KM_PER_DEG
    return None


def snap_to_ladder(resolution_km: float, ladder: list[int]) -> int:
    """
    Finest ladder level whose cell size is >= `resolution_km`. A stream coarser than every
    level goes to the coarsest level; one finer than every level goes to the finest.
    """
    assert len(ladder) >= 1
    chosen = ladder[0]
    for lvl in sorted(ladder):
        if cell_km(lvl) >= resolution_km:
            chosen = lvl
        else:
            break
    return chosen


def assign_stream_levels(cf, ladder: list[int]) -> tuple[dict[str, int], dict[str, list[int]]]:
    """
    Build (encode_level, decode_levels) for every stream in `cf.streams`.

    Encode level: explicit `latent_level` > single-level ladder > snapped resolution >
    error. Decode levels: explicit `decode_levels` > [own level] for the coarsest level >
    [coarsest .. own level].

    For a multi-level ladder every level must be written by at least one stream that has
    source data (non-diagnostic), since each level's encoder needs input tokens.
    """
    ladder = sorted(ladder)
    single = len(ladder) == 1

    encode_level: dict[str, int] = {}
    decode_levels: dict[str, list[int]] = {}

    for name, scfg in cf.streams.items():
        # --- encode level ---
        if "latent_level" in scfg:
            lvl = int(scfg["latent_level"])
            assert lvl in ladder, (
                f"stream {name!r}: latent_level={lvl} is not in the latent level ladder {ladder}."
            )
        elif single:
            lvl = ladder[0]
        else:
            res = _resolution_km(scfg)
            if res is None:
                raise ValueError(
                    f"stream {name!r}: with several latent levels give either `latent_level` "
                    f"or a resolution (`resolution_km` / `resolution_deg`) to snap onto {ladder}."
                )
            lvl = snap_to_ladder(res, ladder)
        encode_level[name] = lvl

        # --- decode levels ---
        if "decode_levels" in scfg:
            dl = sorted({int(x) for x in scfg["decode_levels"]})
            for x in dl:
                assert x in ladder, f"stream {name!r}: decode level {x} is not in {ladder}."
                assert x <= lvl, (
                    f"stream {name!r}: decode level {x} is finer than its encode level {lvl}; "
                    "a stream can only read its own level and coarser ones."
                )
            assert lvl in dl, (
                f"stream {name!r}: decode_levels {dl} must include its encode level {lvl} "
                "(its target coordinates are tokenized on that level's cells)."
            )
            decode_levels[name] = dl
        elif single or lvl == ladder[0]:
            decode_levels[name] = [lvl]
        else:
            decode_levels[name] = [x for x in ladder if x <= lvl]

    if not single:
        for lvl in ladder:
            writers = [
                n
                for n, s in cf.streams.items()
                if encode_level[n] == lvl and not s.get("diagnostic", False)
            ]
            if len(writers) == 0:
                raise ValueError(
                    f"latent level {lvl} has no (non-diagnostic) stream encoding into it; "
                    f"stream encode levels: {encode_level}. Assign a stream to it with "
                    "`latent_level` or remove the level from `latent_levels`."
                )
        for name in cf.streams:
            _logger.info(
                "stream %-12s encode@hl%d  decode@%s",
                name,
                encode_level[name],
                ",".join(f"hl{x}" for x in decode_levels[name]),
            )

    return encode_level, decode_levels
