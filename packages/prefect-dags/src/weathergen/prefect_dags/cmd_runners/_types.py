from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Protocol

from weathergen.prefect_dags.result import Result


@dataclass
class Command:
    command: str | list[str]
    working_directory: str | Path | None = None
    env_vars: dict[str, str] | None = None


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    return_code: int


class CommandRunner(Protocol):
    """
    Interface for running commands, abstracting away the
    location of these commands (e.g. local machine, remote server, etc.).
    """

    name: str

    hpc: str

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        """Run a shell command and return its result, or the error if any.

        This method blocks until the command finishes executing.

        If any error occurs during command execution, it should be captured and returned as an OpError, rather than being raised.

        logger: must be provided. Default loggers do not work well in python
        between asynchronous and multi-threaded contexts, so we require the caller to provide a logger that they know works in their context.
        """
        ...
