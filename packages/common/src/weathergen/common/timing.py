import dataclasses
import datetime
import typing


@dataclasses.dataclass
class Timing:
    mean: float
    std: float
    max: float
    min: float
    n: int

    def as_metric(self):
        pass

    def metric_key(self, timer_name: str):
        return f"perf.timing.{timer_name}"


@dataclasses.dataclass
class Timer:
    name: str
    substeps: dict[str, "Timer"]
    # TODO: use numpy buffers?
    records: list[datetime.datetime]
    _parent: typing.Self | None = None
    _start_time: datetime.datetime | None = None
    _active_substep: typing.Self | None = None

    def record(self):
        if self._parent is not None:
            self._parent._set_active_substep(self)
        else:
            msg = f"Root timer {self.name} should never be used to record"
            raise ValueError(msg)

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
        self.records.append(stop_time - self._start_time)

    def reset(self):
        timings = self.get_result()
        for timer in self.substeps.values():
            timings |= timer.reset()

        self.substeps = {}
        self.records = []
        self._previous_time = None
        return timings

    def get_result(self):
        s = sum(self.records)
        n = len(self.records)
        try:
            timing = Timing(
                s / n,
                sum((record - s / n) ** 2 for record in self.records) ** 0.5,
                max(self.records),
                min(self.records),
                n,
            )
        except ZeroDivisionError:
            timing = None

        return {self.name: timing}

    def _get_labels(self) -> list[str]:
        return self.name.split(".")[1:]  # exclude "root" prefix


# TODO test with multiple MPI ranks
_global_timer = Timer("root", [], [])
_timers = {_global_timer.name: _global_timer}


def record(*labels):
    _get_timer(_global_timer, labels, create=True).record()


def reset(*labels) -> dict[str, Timing]:
    try:
        return _get_timer(_global_timer, labels).reset()
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
