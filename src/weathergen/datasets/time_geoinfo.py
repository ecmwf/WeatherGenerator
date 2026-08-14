# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Recomputation of time-varying geoinfo channels for repeated target coordinates.

When a stream sets ``repeat_steps``, the sampler reads the target window once and the
model reuses that single target-coordinate tensor for every forecast step. That is only
correct for the parts of the tensor that do not depend on time. The geoinfo block does:
``cos/sin_local_time``, ``cos/sin_julian_day`` and ``insolation`` are all functions of the
valid time, so leaving them pinned tells the decoder that every lead time has the same
time of day and the same day of year.

This module recomputes those columns in place on whatever device the coordinate tensor
already lives on, so the memory saving of ``repeat_steps`` is kept: one coordinate tensor
per sample and stream, mutated per forecast step, instead of one tensor per step.

Two things make this less trivial than it looks:

* A target window is not a single instant. With a 6 h window over an hourly dataset each
  window holds six hourly slices, so rows of the coordinate tensor carry several distinct
  valid times. They take only a handful of distinct values though, so each row stores a
  slot index into the window's unique time offsets and the per-slot scalars are computed
  once per forecast step.
* The values in the tensor are normalised, so anything written here has to go through the
  same per-channel statistics the sampler applies in ``collect_datasources`` (see
  ``DataReaderBase.normalize_geoinfos``).

The formulas follow earthkit (``earthkit.data.sources.forcings`` and
``earthkit.meteo.solar``), which is what builds the anemoi datasets these streams read.
``verify_against_reference`` checks them against the values actually stored in the dataset
before enabling a channel, so a dataset built with a different convention degrades to a
frozen channel and a warning rather than to plausible-looking wrong values.
"""

import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)

# Channels this module knows how to reconstruct. Anything else in the geoinfo block is
# either genuinely static (orography, land-sea mask, lat/lon) or unsupported.
TRIG_CHANNELS = (
    "cos_local_time",
    "sin_local_time",
    "cos_julian_day",
    "sin_julian_day",
)
SOLAR_CHANNELS = ("insolation",)
SUPPORTED_CHANNELS = TRIG_CHANNELS + SOLAR_CHANNELS

# Every channel we can rebuild is recomputed by default; verify_against_reference() drops
# the ones whose formula does not reproduce what the dataset stores.
DEFAULT_CHANNELS = SUPPORTED_CHANNELS

# Max absolute difference, in normalised units, for a channel to count as a match. These
# channels are O(1) after normalisation, so this is tight while still tolerating the
# float32 round-trip through the coordinate tensor.
VERIFY_TOLERANCE = 2e-3

# earthkit.meteo.solar.DAYS_PER_YEAR
DAYS_PER_YEAR = 365.25
_SECONDS_PER_DAY = 86400.0

# Layout of the target-coordinate tensor built by get_target_coords_local():
#   col 0                     : stream id
#   cols 1 .. 1+N_TIME        : encoded time within the target window
#   next n_geoinfo cols       : the geoinfo block (this is what we rewrite)
#   remaining N_GEOMETRY cols : healpix local geometry, static
N_TIME_COLS = 5  # width of encode_times_target()
N_GEOMETRY_COLS = 5 * (3 * 5) + 3 * 8
GEOINFO_OFFSET = 1 + N_TIME_COLS


def expected_coords_width(num_geoinfo: int) -> int:
    """Width the target-coordinate tensor must have for GEOINFO_OFFSET to be correct."""
    return 1 + N_TIME_COLS + num_geoinfo + N_GEOMETRY_COLS


@dataclasses.dataclass
class TimeVaryingGeoinfo:
    """Everything needed to rebuild the time-varying geoinfo columns of one stream.

    Attributes
    ----------
    reference_step :
        Forecast step whose target coordinates were read and are reused for every other
        step. Equal to the sampler's ``output_offset``.
    valid_times :
        Window start of every forecast step, indexed by global forecast step so that
        ``valid_times[step]`` lines up with ``StreamData.target_coords`` indexing.
    slot_offsets :
        Seconds between each distinct row time in the window and that window's start.
        A row's valid time at step ``n`` is ``valid_times[n] + slot_offsets[slot]``.
    columns :
        Channel name -> column index into the target-coordinate tensor.
    mean, stdev :
        Normalisation statistics for those channels, taken from the reader so recomputed
        values land in the same space as the ones read from the dataset.
    """

    reference_step: int
    valid_times: npt.NDArray[np.datetime64]
    slot_offsets: npt.NDArray[np.float64]
    columns: dict[str, int]
    mean: dict[str, float]
    stdev: dict[str, float]

    def is_empty(self) -> bool:
        return len(self.columns) == 0


def row_time_slots(
    row_times: npt.NDArray[np.datetime64], window_start: np.datetime64
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """Map each row's valid time onto a small set of offsets from the window start.

    A window holds only as many distinct times as it has data slices (six, for a 6 h
    window over hourly data), so the per-row information collapses to one small index.
    """

    offsets = (
        np.asarray(row_times, dtype="datetime64[s]").astype("int64")
        - np.datetime64(window_start, "s").astype("int64")
    ).astype(np.float64)
    unique, inverse = np.unique(offsets, return_inverse=True)

    return inverse.astype(np.int32), unique


def build_time_varying_geoinfo(
    stream_info: dict,
    reader,
    reference_step: int,
    valid_times: npt.NDArray[np.datetime64],
    slot_offsets: npt.NDArray[np.float64],
    coords_offset: int,
) -> TimeVaryingGeoinfo | None:
    """Collect the metadata needed to recompute a stream's time-varying geoinfos.

    Returns ``None`` when the stream does not repeat its target coordinates, so nothing
    changes for streams that read every forecast step normally.
    """

    if not stream_info.get("repeat_steps", False):
        return None

    geoinfo_channels = list(getattr(reader, "geoinfo_channels", []) or [])
    if not geoinfo_channels:
        return None

    requested = stream_info.get("recompute_geoinfo_channels", DEFAULT_CHANNELS)

    unknown = [c for c in requested if c not in SUPPORTED_CHANNELS]
    if unknown:
        msg = (
            f"recompute_geoinfo_channels contains channels this module cannot rebuild: "
            f"{unknown}. Supported: {list(SUPPORTED_CHANNELS)}."
        )
        raise ValueError(msg)

    mean_geoinfo = np.asarray(reader.mean_geoinfo, dtype=np.float64)
    stdev_geoinfo = np.asarray(reader.stdev_geoinfo, dtype=np.float64)

    columns: dict[str, int] = {}
    mean: dict[str, float] = {}
    stdev: dict[str, float] = {}
    for channel in requested:
        if channel not in geoinfo_channels:
            continue
        idx = geoinfo_channels.index(channel)
        columns[channel] = coords_offset + idx
        mean[channel] = float(mean_geoinfo[idx])
        # matches the guard in DataReaderBase.normalize_geoinfos
        sd = float(stdev_geoinfo[idx])
        stdev[channel] = 1.0 if np.isclose(sd, 0.0) else sd

    if not columns:
        return None

    return TimeVaryingGeoinfo(
        reference_step=reference_step,
        valid_times=np.asarray(valid_times),
        slot_offsets=np.asarray(slot_offsets, dtype=np.float64),
        columns=columns,
        mean=mean,
        stdev=stdev,
    )


def _julian_day(times: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]:
    """Days elapsed since 1 January of each time's own year.

    Zero-based, matching ``earthkit.meteo.solar.julian_day`` (and the ``julian_day``
    helper in ``earthkit.data.sources.forcings``).
    """
    times = np.asarray(times, dtype="datetime64[s]")
    year_start = times.astype("datetime64[Y]").astype("datetime64[s]")
    return (times - year_start).astype("float64") / _SECONDS_PER_DAY


def _hours_since_midnight(times: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]:
    """Fractional hours since midnight, as used by earthkit's ``local_time``."""
    times = np.asarray(times, dtype="datetime64[s]")
    day_start = times.astype("datetime64[D]").astype("datetime64[s]")
    return (times - day_start).astype("float64") / 3600.0


def _solar_declination_angle(
    times: npt.NDArray[np.datetime64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Declination [degrees] and time correction [h.degrees].

    Verbatim from ``earthkit.meteo.solar.solar_declination_angle``.
    """
    angle = _julian_day(times) / DAYS_PER_YEAR * np.pi * 2

    declination = (
        0.396372
        - 22.91327 * np.cos(angle)
        + 4.025430 * np.sin(angle)
        - 0.387205 * np.cos(2 * angle)
        + 0.051967 * np.sin(2 * angle)
        - 0.154527 * np.cos(3 * angle)
        + 0.084798 * np.sin(3 * angle)
    )
    time_correction = (
        0.004297
        + 0.107029 * np.cos(angle)
        - 1.837877 * np.sin(angle)
        - 0.837378 * np.cos(2 * angle)
        - 2.340475 * np.sin(2 * angle)
    )
    return declination, time_correction


def compute_channel(
    channel: str,
    times: npt.NDArray[np.datetime64],
    slot: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
) -> torch.Tensor:
    """Raw (unnormalised) value of one time-varying geoinfo channel, per row.

    ``times`` holds the distinct valid times present in the window, ``slot`` maps each row
    onto one of them, and ``lat``/``lon`` are degrees on the device the result lands on.
    """

    device = lon.device

    def per_row(values: npt.NDArray[np.float64]) -> torch.Tensor:
        """Broadcast a per-distinct-time scalar out to every row."""
        table = torch.as_tensor(values, dtype=torch.float64, device=device)
        return table[slot]

    if channel in ("cos_julian_day", "sin_julian_day"):
        # spatially constant: a function of the date alone
        angle = per_row(_julian_day(times) / DAYS_PER_YEAR * np.pi * 2)
        return torch.cos(angle) if channel.startswith("cos") else torch.sin(angle)

    if channel in ("cos_local_time", "sin_local_time"):
        # earthkit: (lon / 360 * 24 + hours_since_midnight) % 24, then /24 * 2pi
        local_time = lon.double() / 360.0 * 24.0 + per_row(_hours_since_midnight(times))
        angle = torch.remainder(local_time, 24.0) / 24.0 * np.pi * 2
        return torch.cos(angle) if channel.startswith("cos") else torch.sin(angle)

    if channel == "insolation":
        # earthkit.data maps insolation onto the cosine of the solar zenith angle
        declination, time_correction = _solar_declination_angle(times)
        dec = torch.deg2rad(per_row(declination))
        # note: earthkit uses the integer hour here, not the fractional one it uses for
        # local_time, and adds longitude in degrees rather than hours
        hour = per_row(_integer_hour(times))
        lat_rad = torch.deg2rad(lat.double())
        solar_angle = torch.deg2rad((hour - 12.0) * 15.0 + lon.double() + per_row(time_correction))
        zenith = torch.sin(dec) * torch.sin(lat_rad) + torch.cos(dec) * torch.cos(
            lat_rad
        ) * torch.cos(solar_angle)
        return torch.clamp(zenith, min=0.0)

    msg = f"Cannot recompute geoinfo channel {channel!r}."
    raise ValueError(msg)


def _integer_hour(times: npt.NDArray[np.datetime64]) -> npt.NDArray[np.float64]:
    """Whole-hour component of each time, as ``datetime.hour`` would give."""
    times = np.asarray(times, dtype="datetime64[s]")
    day_start = times.astype("datetime64[D]").astype("datetime64[s]")
    return np.floor((times - day_start).astype("float64") / 3600.0)


def _row_times(meta: TimeVaryingGeoinfo, fstep: int) -> npt.NDArray[np.datetime64]:
    """Distinct valid times covered by the window of forecast step ``fstep``."""
    base = np.datetime64(meta.valid_times[fstep], "s")
    return base + meta.slot_offsets.astype("timedelta64[s]")


def recompute_geoinfos(
    target_coords: torch.Tensor,
    latlon: torch.Tensor,
    slot: torch.Tensor,
    meta: TimeVaryingGeoinfo,
    fstep: int,
) -> None:
    """Rewrite the time-varying geoinfo columns of ``target_coords`` for ``fstep``.

    Mutates ``target_coords`` in place on its current device.
    """

    if meta.is_empty() or target_coords.numel() == 0:
        return

    if fstep >= len(meta.valid_times):
        msg = (
            f"No valid time recorded for forecast step {fstep}; "
            f"only {len(meta.valid_times)} steps are known."
        )
        raise IndexError(msg)

    if latlon.shape[0] != target_coords.shape[0] or slot.shape[0] != target_coords.shape[0]:
        msg = (
            f"lat/lon has {latlon.shape[0]} rows and slot has {slot.shape[0]}, but the "
            f"target coordinates have {target_coords.shape[0]}; all must correspond row "
            "for row."
        )
        raise ValueError(msg)

    times = _row_times(meta, fstep)
    lat, lon = latlon[:, 0], latlon[:, 1]
    slot = slot.long()

    for channel, column in meta.columns.items():
        raw = compute_channel(channel, times, slot, lat, lon)
        # same normalisation the reader applies to the values this replaces
        normalised = (raw - meta.mean[channel]) / meta.stdev[channel]
        target_coords[:, column] = normalised.to(target_coords.dtype)


def verify_against_reference(
    meta: TimeVaryingGeoinfo,
    target_coords: torch.Tensor,
    latlon: torch.Tensor,
    slot: torch.Tensor,
    stream_name: str,
    tolerance: float = VERIFY_TOLERANCE,
) -> TimeVaryingGeoinfo:
    """Drop any channel whose formula does not reproduce the dataset's own values.

    At the reference step the coordinate tensor still holds the geoinfos as they were read
    from the dataset, so it doubles as ground truth. A channel that matches is safe to
    rewrite at every other step; one that does not means this module's convention
    disagrees with whatever built the dataset, and rewriting it would feed the decoder a
    plausible but wrong field.

    The metadata is always returned, restricted to the channels that verified. It also
    carries ``reference_step``, which is what tells the model that this stream's
    coordinates are shared across forecast steps -- dropping it would make the model index
    coordinates per step, and a repeating stream only holds the reference step, so every
    later step would find nothing and silently produce no predictions.
    """

    if meta.is_empty() or target_coords.numel() == 0:
        return meta

    times = _row_times(meta, meta.reference_step)
    lat, lon = latlon[:, 0], latlon[:, 1]
    slot_l = slot.long()

    verified: dict[str, int] = {}
    rejected: dict[str, float] = {}
    for channel, column in meta.columns.items():
        raw = compute_channel(channel, times, slot_l, lat, lon)
        normalised = (raw - meta.mean[channel]) / meta.stdev[channel]
        diff = (normalised.to(target_coords.dtype) - target_coords[:, column]).abs().max().item()
        if diff <= tolerance:
            verified[channel] = column
        else:
            rejected[channel] = diff

    if rejected:
        logger.warning(
            "Stream %s: recomputed geoinfo channels %s do not match the values stored in "
            "the dataset (max normalised difference %s, tolerance %g). They will stay "
            "FROZEN at forecast step %d, which is wrong for any rollout longer than one "
            "step. The formula in weathergen/datasets/time_geoinfo.py does not match this "
            "dataset's convention for them; run scripts/check_time_geoinfo.py to compare.",
            stream_name,
            sorted(rejected),
            {c: f"{d:.3g}" for c, d in sorted(rejected.items())},
            tolerance,
            meta.reference_step,
        )

    if not verified:
        # coordinates are still repeated, they just keep the geoinfos they were read with
        return dataclasses.replace(meta, columns={}, mean={}, stdev={})

    logger.info(
        "Stream %s: recomputing time-varying geoinfo channels %s per forecast step "
        "(verified against the dataset at step %d).",
        stream_name,
        sorted(verified),
        meta.reference_step,
    )

    return dataclasses.replace(
        meta,
        columns=verified,
        mean={c: meta.mean[c] for c in verified},
        stdev={c: meta.stdev[c] for c in verified},
    )
