# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``evaluate-mpi``: the evaluation under MPI.

Same command line as ``evaluate``.  With more than one rank, ``distributed.enabled: auto``
(the default) or ``true`` runs the chunked evaluation with rank 0 as master and the other
ranks as workers (``distributed.runner``); ``false`` runs the serial evaluation on rank 0.
Needs the ``mpi`` extra (``mpi4py``).

Run the ranks as the venv's interpreter directly, with the environment synced beforehand::

    srun --mpi=pmix .venv/bin/python -m weathergen.evaluate.distributed.mpi --config cfg.yml

not through ``uv run``: inside a job it would rebuild the editable packages from every rank
at once, and it does not let go of a rank that leaves with ``os._exit`` (the ranks end up
as zombies under ``uv`` and the step never ends).
"""

import logging
import sys

from mpi4py import MPI
from omegaconf import open_dict

from weathergen.common.logger import init_loggers
from weathergen.evaluate.distributed.runner import run_mpi
from weathergen.evaluate.run_evaluation import (
    config_from_args,
    evaluate_from_config,
    run_chunked,
)

_logger = logging.getLogger(__name__)


class MpiComm:
    """``MPI.COMM_WORLD`` as the communicator ``distributed.runner`` expects."""

    def __init__(self, comm):
        self._comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    def send(self, obj, dest: int, tag: int) -> None:
        self._comm.send(obj, dest=dest, tag=tag)

    def recv(self) -> tuple[object, int, int]:
        status = MPI.Status()
        obj = self._comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        return obj, status.Get_source(), status.Get_tag()


def main() -> None:
    """Entry point of ``evaluate-mpi``."""
    init_loggers()
    cfg, mlflow_client = config_from_args(sys.argv[1:])
    comm = MpiComm(MPI.COMM_WORLD)

    enabled = (cfg.get("distributed") or {}).get("enabled", "auto")
    if enabled == "auto":
        enabled = comm.size > 1
    with open_dict(cfg):
        if not cfg.get("distributed"):
            cfg.distributed = {}
        cfg.distributed.enabled = bool(enabled)

    if not enabled:
        if comm.rank == 0:
            _logger.info("distributed.enabled is false: serial evaluation on rank 0.")
            evaluate_from_config(cfg, mlflow_client)
        return
    if comm.size == 1:
        _logger.warning("One MPI rank: running the chunks in this process.")
        run_chunked(cfg, mlflow_client)
        return
    status = 0
    try:
        run_mpi(cfg, mlflow_client, comm)
    except Exception:  # noqa: BLE001 - logged in full; the exit status carries it
        _logger.exception(f"MPI rank {comm.rank}/{comm.size}: the evaluation failed.")
        status = 1
    _logger.info(f"MPI rank {comm.rank}/{comm.size}: finalising MPI.")
    MPI.Finalize()
    _logger.info(f"MPI rank {comm.rank}/{comm.size}: MPI finalised, leaving.")
    sys.exit(status)


if __name__ == "__main__":
    main()
