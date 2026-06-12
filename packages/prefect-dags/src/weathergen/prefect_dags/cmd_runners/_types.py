import re
import shlex
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Protocol

from weathergen.prefect_dags.result import Result

# A leading tilde component: "~" or "~user".
_TILDE_PREFIX_RE = re.compile(r"~[A-Za-z0-9._-]*")


def quote_path(path: str | Path) -> str:
    """Shell-quote a path for a remote shell, preserving tilde expansion.

    `shlex.quote("~/work")` returns `'~/work'` (single-quoted), and quoting
    suppresses the shell's tilde expansion — `cd '~/work'` then fails with
    "No such file or directory". Tilde expansion only happens when the tilde
    is unquoted at the start of a word, so leave a leading `~` or `~user`
    component unquoted and quote only the remainder of the path.
    """
    s = str(path)
    m = _TILDE_PREFIX_RE.match(s)
    if m is None:
        return shlex.quote(s)
    prefix, rest = s[: m.end()], s[m.end() :]
    if rest in ("", "/"):
        return s
    if not rest.startswith("/"):
        # Not a tilde *component* (e.g. "~weird~name"): quote the whole thing.
        return shlex.quote(s)
    # Keep the slash unquoted with the prefix so the tilde component ends
    # cleanly before any quoting starts: `~/'a b'`, `~user/'a b'`.
    return f"{prefix}/{shlex.quote(rest[1:])}"


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

        If any error occurs during command execution, it should be captured and returned as an
        OpError, rather than being raised.

        logger: must be provided. Default loggers do not work well in python
        between asynchronous and multi-threaded contexts, so we require the caller to provide a
        logger that they know works in their context.
        """
        ...
