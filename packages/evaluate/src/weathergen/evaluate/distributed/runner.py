# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The master / worker run of the chunked evaluation over a communicator.

Rank 0 drives the stages of every (run, stream) (``run_evaluation.stream_stages``) at once:
the tasks of all batches that are ready go into one queue and are handed out to the workers
as they ask for work, so a long chunk of one run does not hold the others back.  A batch's
stage is advanced when its last task is in.  The workers only call what they are given.

The communicator is anything with ``rank``, ``size``, ``send(obj, dest, tag)`` and
``recv() -> (obj, source, tag)``: ``distributed.mpi`` wraps ``MPI.COMM_WORLD``, the tests
use queues between threads.
"""

import enum
import logging
import os
import socket
import traceback
from pathlib import Path

import matplotlib
from joblib.externals.loky import get_reusable_executor
from mlflow.client import MlflowClient

from weathergen.evaluate.distributed.chunking import parse_distributed_config
from weathergen.evaluate.run_evaluation import (
    accumulate_stream_results,
    build_stream_tasks,
    finalise,
    stream_stages,
)

_logger = logging.getLogger(__name__)


class Tag(enum.IntEnum):
    """Message tags: a worker sends ``READY`` or a ``RESULT``, the master a ``TASK`` or ``STOP``."""

    READY = 1
    TASK = 2
    RESULT = 3
    STOP = 4


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
    try:
        if comm.rank == 0:
            master(cfg, mlflow_client, comm)
        else:
            worker(comm)
    finally:
        # Shut the loky worker processes of this rank (scoring, plotting) down here, in an
        # orderly way: they then release joblib's temporary memmap folders themselves, instead
        # of leaving them to the resource tracker at interpreter exit.
        get_reusable_executor().shutdown(wait=True)
        _logger.info(f"MPI rank {comm.rank}/{comm.size}: done, exiting.")
        logging.getLogger().removeHandler(handler)


def master(cfg: dict, mlflow_client: MlflowClient | None, comm) -> None:
    """Hand out the tasks of all (run, stream) stages, then ``finalise``.

    A task that fails on a worker, or a stage that fails here, fails its (run, stream): the
    rest of its stages are skipped, the other (run, stream) go on, and the failures are
    raised together at the end, after ``finalise`` of what succeeded.
    """
    _, sample_size, fstep_size = parse_distributed_config(cfg)
    tasks = build_stream_tasks(cfg)
    names = {i: f"RUN {task['run_id']} - {task['stream']}" for i, task in enumerate(tasks)}
    stages = {i: stream_stages(task, sample_size, fstep_size) for i, task in enumerate(tasks)}
    queue: list[tuple] = []  # (stream, task index, function, kwargs, description)
    pending: dict[int, list] = {}  # stream -> the outputs of its current batch
    missing: dict[int, int] = {}  # stream -> how many of them are still to come
    results: dict[int, tuple] = {}
    failed: dict[int, str] = {}

    def advance(i: int, outputs: list | None) -> None:
        """Send a batch's outputs to stage generator *i* and queue its next batch."""
        try:
            batch = stages[i].send(outputs)
        except StopIteration as done:
            results[i] = done.value
            return
        except Exception:  # noqa: BLE001 - the error is reported, the other streams go on
            failed[i] = traceback.format_exc()
            _logger.error(f"{names[i]}: failed:\n{failed[i]}")
            return
        if not batch:
            advance(i, [])
            return
        pending[i] = [None] * len(batch)
        missing[i] = len(batch)
        queue.extend((i, j, *task) for j, task in enumerate(batch))

    for i in stages:
        advance(i, None)

    idle: list[int] = []
    busy: set[int] = set()  # the ranks that hold a task
    running = set(range(1, comm.size))  # the ranks not told to stop yet
    try:
        while running:
            msg, source, tag = comm.recv()
            busy.discard(source)
            if tag == Tag.READY:
                _logger.info(f"Worker rank {source} ready on {msg}.")
            elif tag == Tag.RESULT:
                i, j, output, error = msg
                if i in failed:
                    pass  # a task of a stream that already failed elsewhere
                elif error is not None:
                    failed[i] = error
                    pending.pop(i, None)
                    missing.pop(i, None)
                    stages[i].close()
                    _logger.error(f"{names[i]}: task {j + 1} failed on rank {source}:\n{error}")
                else:
                    pending[i][j] = output  # may be None: finish_stream returns nothing
                    missing[i] -= 1
                    if missing[i] == 0:
                        outputs = pending.pop(i)
                        del missing[i]
                        advance(i, outputs)
            idle.append(source)
            while idle and queue:
                i, j, fn, kwargs, description = queue.pop(0)
                if i in failed:
                    continue
                rank = idle.pop(0)
                _logger.info(f"{description} -> rank {rank}")
                comm.send((i, j, fn, kwargs, description), rank, Tag.TASK)
                busy.add(rank)
            if not queue and not pending:
                for rank in idle:
                    comm.send(None, rank, Tag.STOP)
                    running.discard(rank)
                idle.clear()
    except Exception:
        # Stop the workers before leaving: left waiting for a task, they would keep the job
        # alive until its time limit.  The busy ones are stopped once their result is in,
        # which their send waits for.
        _logger.exception("The master failed; stopping the workers.")
        for rank in running - busy:
            comm.send(None, rank, Tag.STOP)
        while busy:
            _, source, _ = comm.recv()
            busy.discard(source)
            comm.send(None, source, Tag.STOP)
        raise

    _logger.info(
        "All chunk tasks done, workers stopped; finalising on the master: MLflow upload "
        "and the verification plots across runs."
    )
    finalise(cfg, *accumulate_stream_results([results[i] for i in sorted(results)]), mlflow_client)
    _logger.info("Evaluation finished.")
    if failed:
        raise RuntimeError(f"{len(failed)} (run, stream) failed:\n" + "\n".join(failed.values()))


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
