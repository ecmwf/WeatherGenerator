# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The master / worker run of the chunked evaluation over a communicator.

Rank 0 is the master.  It runs ``run_evaluation._run_stream_in_chunks`` of every (run, stream)
on a thread of its own, and the batches those threads ask for go to the workers through the
main thread, which is the only thread that talks to them: the tasks of all batches that are
ready are handed out to the workers as they ask for work, so a long chunk of one run does
not hold the others back.  The workers only call what they are given.

The communicator is anything with ``rank``, ``size``, ``send(obj, dest, tag)`` and
``recv() -> (obj, source, tag)`` that may be used from two threads at once (one receiving,
one sending): ``distributed.mpi`` wraps ``MPI.COMM_WORLD``, the tests use queues between
threads.
"""

import enum
import logging
import os
import queue
import socket
import threading
import traceback
from pathlib import Path

import matplotlib
from mlflow.client import MlflowClient

from weathergen.evaluate.distributed.chunking import parse_distributed_config
from weathergen.evaluate.run_evaluation import (
    _run_stream_in_chunks,
    accumulate_stream_results,
    build_stream_tasks,
    finalise,
)

_logger = logging.getLogger(__name__)


class Tag(enum.IntEnum):
    """Message tags: a worker sends ``READY`` or a ``RESULT``, the master a ``TASK`` or ``STOP``.

    ``BATCH`` and ``FINISHED`` never go over the wire: they are what the master's stream
    threads put in the main thread's inbox.
    """

    READY = 1
    TASK = 2
    RESULT = 3
    STOP = 4
    BATCH = 5
    FINISHED = 6


def run_mpi(cfg: dict, mlflow_client: MlflowClient | None, comm) -> None:
    """``run_chunked`` over *comm*: rank 0 is the master, the others are workers."""
    matplotlib.use("Agg")
    log_dir = Path(cfg.evaluation.get("summary_dir", ".")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # one set of files per run: the Slurm job and step ids (a new step per srun in an
    # allocation), or the rank's pid outside Slurm
    job = ".".join(filter(None, (os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID"))))
    log_file = log_dir / f"evaluation_{job or f'pid{os.getpid()}'}_rank{comm.rank}.log"
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)  # the root logger lets DEBUG through to its handlers
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s"))
    logging.getLogger().addHandler(handler)
    role = "master" if comm.rank == 0 else "worker"
    _logger.info(
        f"MPI rank {comm.rank}/{comm.size} ({role}) on {socket.gethostname()}: "
        f"logging to the console and to {log_file}"
    )
    if comm.rank == 0:
        master(cfg, mlflow_client, comm)
    else:
        worker(comm)


class Stream:
    """One (run, stream) on the master: its stages run on a thread of their own.

    The thread only sees the main thread's *inbox*: ``run_tasks`` puts each batch there and
    waits for the outputs, which the main thread fills in with ``on_result`` as the workers
    return them, and ``run`` puts ``FINISHED`` there at the end.
    """

    def __init__(self, index: int, task: dict, inbox: queue.Queue):
        self.index = index  # the position in the master's list; the tasks carry it
        self.name = f"RUN {task['run_id']} - {task['stream']}"
        self.task = task
        self.inbox = inbox
        self.outputs: list = []  # of the batch under way
        self.missing = 0  # how many of them are still to come
        self.complete = threading.Event()  # none missing, or the stream failed
        self.result: tuple | None = None  # what _run_stream_in_chunks returned
        self.error: str | None = None  # the traceback that failed the stream

    def run(self, sample_size: int | None, fstep_size: int | None) -> None:
        """The thread's body: the stages, with their batches run on the workers."""
        try:
            self.result = _run_stream_in_chunks(self.task, sample_size, fstep_size, self.run_tasks)
        except Exception:  # noqa: BLE001 - the failure is reported, the other streams go on
            if self.error is None:  # a task's failure on a worker is already logged and kept
                self.error = traceback.format_exc()
                _logger.error(f"{self.name}: failed:\n{self.error}")
        self.inbox.put((Tag.FINISHED, self, None))

    def run_tasks(self, tasks: list[tuple]) -> list:
        """``_run_stream_in_chunks``'s *run_tasks*: hand the batch over, wait for its outputs."""
        if not tasks:
            return []
        self.outputs, self.missing = [None] * len(tasks), len(tasks)
        self.complete.clear()
        self.inbox.put((Tag.BATCH, self, [(self.index, j, *task) for j, task in enumerate(tasks)]))
        self.complete.wait()
        if self.error is not None:
            raise RuntimeError(f"{self.name}: a task of the batch failed on a worker")
        return self.outputs

    def on_result(self, j: int, output, error: str | None, source: int) -> None:
        """Main thread: record task *j*'s result, wake the thread once the batch is complete."""
        if self.error is not None:
            return  # a task of a batch that was abandoned
        if error is not None:
            self.error = error
            _logger.error(f"{self.name}: task {j + 1} failed on rank {source}:\n{error}")
        self.outputs[j] = output  # may be None: finish_stream returns nothing
        self.missing -= 1
        if self.missing == 0 or error is not None:
            self.complete.set()


class Workers:
    """The worker ranks as the master sees them: idle, holding a task, or stopped.

    ``receive`` is the master's receiver thread: every message from a worker goes into the
    main thread's *inbox*, until the master stops the receiver like a worker (rank 0).
    """

    def __init__(self, comm, inbox: queue.Queue):
        self.comm = comm
        self.inbox = inbox
        self.idle: list[int] = []
        self.busy: set[int] = set()
        self.running = set(range(1, comm.size))  # not told to stop yet
        self.tasks: list[tuple] = []  # waiting for a worker

    def receive(self) -> None:
        msg, source, tag = self.comm.recv()
        while tag != Tag.STOP:
            self.inbox.put((tag, source, msg))
            msg, source, tag = self.comm.recv()

    def free(self, rank: int) -> None:
        self.busy.discard(rank)
        self.idle.append(rank)

    def dispatch(self, streams: list[Stream]) -> None:
        """Hand the waiting tasks to the idle workers; those of a failed stream are dropped."""
        while self.idle and self.tasks:
            task = self.tasks.pop(0)
            if streams[task[0]].error is None:
                rank = self.idle.pop(0)
                _logger.info(f"{task[4]} -> rank {rank}")
                self.comm.send(task, rank, Tag.TASK)
                self.busy.add(rank)

    def stop(self, ranks) -> None:
        for rank in list(ranks):
            self.comm.send(None, rank, Tag.STOP)
            self.running.discard(rank)
        self.idle = [r for r in self.idle if r in self.running]

    def stop_all(self) -> None:
        """Stop every worker: the idle ones now, the busy ones once their result is in."""
        self.stop(self.running - self.busy)
        while self.busy:
            tag, source, _ = self.inbox.get()
            if tag == Tag.RESULT:
                self.busy.discard(source)
                self.stop([source])
        self.stop([0])  # the receiver thread


def master(cfg: dict, mlflow_client: MlflowClient | None, comm) -> None:
    """Run the streams on threads and their batches on the workers, then ``finalise``.

    A task that fails on a worker, or a stage that fails on its thread, fails its (run,
    stream): the others go on, and the failures are raised together at the end, after
    ``finalise`` of what succeeded.  A failure here stops the workers before it is raised:
    left waiting for a task, they would keep the job alive until its time limit.
    """
    _, sample_size, fstep_size = parse_distributed_config(cfg)
    inbox: queue.Queue = queue.Queue()  # all the main thread waits for
    streams = [Stream(i, task, inbox) for i, task in enumerate(build_stream_tasks(cfg))]
    workers = Workers(comm, inbox)
    threading.Thread(target=workers.receive, daemon=True).start()
    for stream in streams:
        threading.Thread(target=stream.run, args=(sample_size, fstep_size), daemon=True).start()
    unfinished = len(streams)
    try:
        while workers.running:
            tag, source, msg = inbox.get()
            if tag == Tag.BATCH:
                workers.tasks.extend(msg)
            elif tag == Tag.FINISHED:
                unfinished -= 1
            elif tag == Tag.READY:
                _logger.info(f"Worker rank {source} ready on {msg}.")
                workers.free(source)
            elif tag == Tag.RESULT:
                workers.free(source)
                streams[msg[0]].on_result(*msg[1:], source)
            workers.dispatch(streams)
            if unfinished == 0 and not workers.tasks:
                workers.stop(workers.idle)
    except Exception:
        _logger.exception("The master failed; stopping the workers.")
        workers.stop_all()
        raise
    workers.stop([0])  # the receiver thread

    _logger.info(
        "All chunk tasks done, workers stopped; finalising on the master: MLflow upload "
        "and the verification plots across runs."
    )
    done = [s.result for s in streams if s.error is None]
    finalise(cfg, *accumulate_stream_results(done), mlflow_client)
    _logger.info("Evaluation finished.")
    failed = [s.error for s in streams if s.error is not None]
    if failed:
        raise RuntimeError(f"{len(failed)} (run, stream) failed:\n" + "\n".join(failed))


def worker(comm) -> None:
    """Call the tasks the master sends until it says stop."""
    comm.send(socket.gethostname(), 0, Tag.READY)
    while True:
        msg, _, tag = comm.recv()
        if tag == Tag.STOP:
            return
        i, j, fn, kwargs, description = msg
        _logger.info(f"Starting: {description}")
        try:
            output, error = fn(**kwargs), None
        except Exception:  # noqa: BLE001 - reported to the master, which decides
            output, error = None, traceback.format_exc()
        _logger.info(f"{'Failed' if error else 'Done'}: {description}")
        comm.send((i, j, output, error), 0, Tag.RESULT)
