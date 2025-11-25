import dataclasses
import datetime
import logging
import typing

import numpy as np

NPDT64 = np.datetime64
NPTDel64 = np.timedelta64

MAX_RECORDS = 200

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Timing:
    """Statistics in ns."""

    mean: float
    std: float
    max: int
    min: int
    n: int

    def as_metric(self, timer_name: str) -> dict[str, int | float]:
        return {
            self.metric_key(timer_name, metric): value
            for metric, value in dataclasses.asdict(self).items()
        }

    @staticmethod
    def metric_key(timer_name: str, metric_name: str | None) -> str:
        key = f"perf.timing.{timer_name}"
        if metric_name is not None:
            key += f".{metric_name}"

        return key


@dataclasses.dataclass
class Timer:
    name: str
    substeps: dict[str, "Timer"]
    # TODO: use numpy buffers?
    records: list[NPTDel64]
    _parent: typing.Self | None = None
    _start_time: datetime.datetime | None = None
    _active_substep: typing.Self | None = None

    def record(self, *labels: str):
        if len(labels) == 0:
            if self._parent is not None:
                self._parent._set_active_substep(self)
            else:
                msg = f"Root timer {self.name} should never be used to record"
                raise ValueError(msg)
        else:
            subtimer = _get_timer(self, *labels, create=True)
            subtimer.record()

    def _set_active_substep(self, substep_timer: "Timer"):
        assert substep_timer.name in self.substeps.keys()

        # TODO figure out best way to record time
        time = datetime.datetime.now()
        if self._active_substep is not None:
            self._active_substep._stop_recording(time)

        self._active_substep = substep_timer
        substep_timer._start_recording(time)

    def _start_recording(self, start_time: datetime.datetime):
        self._start_time = start_time

    def _stop_recording(self, stop_time: datetime.datetime):
        self.records.append(NPTDel64(stop_time - self._start_time))

    def reset(self) -> dict[str, Timing]:
        _logger.debug(f"resetting timer: {self.name}")
        timings = self.get_result()
        for timer in self.substeps.values():
            timings |= timer.reset()

        self._active_substep = None
        self.substeps = {}
        self.records = []
        self._previous_time = None
        return timings

    def get_result(self):
        records = np.array(self.records, dtype="datetime64[ns]").astype(np.int64)
        if records.size != 0:
            timing = Timing(
                records.mean(),
                records.std(),
                records.max(),
                records.min(),
                records.size,
            )
        else:
            timing = Timing(np.nan, np.nan, np.nan, np.nan, 0)

        return {self.name: timing}

    def _get_labels(self) -> list[str]:
        return self.name.split(".")[1:]  # exclude "root" prefix


def record(*labels):
    _logger.debug(f"record timer: {labels}")
    _get_timer(_global_timer, *labels, create=True).record()


def reset(*labels) -> dict[str, Timing]:
    _logger.debug(f"reset timer: {labels}")
    try:
        return _get_timer(_global_timer, *labels).reset()
    except ValueError:
        return {}


def _get_timer(root_timer, *labels: str, create=False) -> Timer:
    timer = root_timer
    for suffix in labels:
        name = f"{timer.name}.{suffix}"
        try:
            timer = timer.substeps[name]
        except KeyError as e:
            if create:
                new_timer = Timer(name, {}, [], _parent=timer)
                timer.substeps[name] = new_timer
                timer = new_timer
            else:
                msg = f"No such timer: {name}"
                raise ValueError(msg) from e

    return timer


# TODO test with multiple MPI ranks
_global_timer = Timer("root", {}, [])

"""This timer should be used within the training loop."""
train: Timer = _get_timer(_global_timer, "train", create=True)

"""This timer should be used within the validation loop."""
validate: Timer = _get_timer(_global_timer, "validate", create=True)
