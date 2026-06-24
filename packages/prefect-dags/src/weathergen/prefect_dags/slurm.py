"""
Utilities for running commands on a Slurm cluster using Prefect.

On purpose, this code does not depend on prefect itself (no task, etc) to
allow portability.

Also, it does not depend on a specific HPC environment or command runner.
The only primitives are calling sbatch and sacct.
"""

import asyncio
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from weathergen.prefect_dags.cmd_runners import (
    CmdContext,
    Command,
    CommandResult,
    run_cmd,
    slurm_account,
)
from weathergen.prefect_dags.cmd_runners._types import quote_path
from weathergen.prefect_dags.result import OpError, Result, is_err


class SubmissionError(Exception):
    """
    Raised when `sbatch` ran to completion but its output did not contain
    an extractable Slurm job id.
    """

    def __init__(self, result: CommandResult, message: str | None = None):
        self.result = result
        detail = message or "Could not parse a Slurm job id from sbatch output"
        super().__init__(
            f"{detail} "
            f"(return_code={result.return_code}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r})"
        )


@dataclass
class SlurmJob:
    """
    Represents a Slurm job to be submitted.
    """

    job_name: str
    # Stdout file (sbatch --output). May contain Slurm filename patterns like %j.
    stdout: str | Path | None = None
    # Stderr file (sbatch --error). May contain Slurm filename patterns like %j.
    stderr: str | Path | None = None
    script_path: str | Path | None = None
    command: str | list[str] | None = None
    # Working directory of the *job* on the compute node (sbatch --chdir).
    working_directory: str | Path | None = None
    # Working directory of the sbatch *submission* process itself.
    submission_directory: str | Path | None = None
    # Walltime limit (sbatch --time). Slurm formats: "MM", "MM:SS", "HH:MM:SS",
    # "D-HH", "D-HH:MM", or "D-HH:MM:SS". The job is killed when this elapses.
    time_limit: str | None = None
    slurm_options: dict[str, str] | None = None
    # Slurm account to charge (sbatch --account).
    # - if None, no --account argument is passed and the cluster default is used.
    # - if "auto", the account is inferred from the context (if supported) or None if unsupported.
    # - if any other string, that value is passed verbatim as --account.
    account: str | Literal["auto"] | None = "auto"


type SlurmJobId = str


@dataclass
class SlurmSubmissionResult:
    """
    Result of submitting a Slurm job, containing the job ID
    and the resolved output paths.
    """

    # The job id
    # Represented as string to ensure JSON serialization does not overflow to floating point.
    job_id: SlurmJobId
    # The stdout file path (sbatch --output), with %j resolved to the job id.
    # TODO: check with the deserialization it stays a Path
    stdout: Path
    # The stderr file path (sbatch --error), with %j resolved to the job id.
    stderr: Path


SlurmJobState = Literal[
    # base states
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PENDING",
    "PREEMPTED",
    "RUNNING",
    "SUSPENDED",
    "TIMEOUT",
    # state flags (reported in place of the base state when set)
    "COMPLETING",
    "CONFIGURING",
    "EXPEDITING",
    "LAUNCH_FAILED",
    "POWER_UP_NODE",
    "RECONFIG_FAIL",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RESV_DEL_HOLD",
    "REVOKED",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
    "UPDATE_DB",
]

_KNOWN_STATES: frozenset[str] = frozenset(get_args(SlurmJobState))


_TERMINAL_STATES: frozenset[SlurmJobState] = frozenset(
    {
        # Slurm's canonical JOB_END set (slurm.h): the job has a final disposition
        # and will not transition further.
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
        # Federation: a sibling cluster cancelled the job — it will not run here.
        "REVOKED",
        # Launch failed permanently.
        "LAUNCH_FAILED",
    }
)

# A path that starts with "~" or "~user" and is resolved by the remote shell.
_TILDE_ABSOLUTE_RE = re.compile(r"^~[A-Za-z0-9._-]*(?:/|$)")


def _is_absolute_or_remote_home_path(path: str | Path) -> bool:
    path_str = str(path)
    return Path(path_str).is_absolute() or _TILDE_ABSOLUTE_RE.match(path_str) is not None


def is_terminal_state(state: SlurmJobState) -> bool:
    """Return True if the given Slurm state is terminal (job will not transition further).

    Note: COMPLETING and STAGE_OUT are NOT terminal — the user's process has
    ended but Slurm is still finalizing; they transition to COMPLETED. Callers
    that poll should keep polling on those.
    """
    return state in _TERMINAL_STATES


@dataclass
class SlurmJobInfo:
    job_id: SlurmJobId
    state: SlurmJobState


async def get_slurm_job_states(
    ctx: CmdContext,
    slurm_job_ids: list[SlurmJobId],
    logger: logging.Logger,
) -> Result[list[SlurmJobInfo]]:
    """Look up the current Slurm state for each job id via `sacct`.

    Raises RuntimeError if sacct itself fails, ValueError on an unknown
    state string, and LookupError if any input id has no sacct record
    (e.g. purged from accounting, or never submitted).
    """
    if not slurm_job_ids:
        return []

    # -X : only the main allocation row, skip .batch / .extern / .<step>
    # -n : no header
    # -P : parsable, '|'-delimited, no padding
    cmd = Command(
        command=[
            "sacct",
            "-j",
            ",".join(slurm_job_ids),
            "-o",
            "JobID,State",
            "-X",
            "-n",
            "-P",
        ]
    )
    res = await run_cmd(ctx, cmd, logger)
    if is_err(res):
        return OpError(err=RuntimeError(f"sacct command failed: {res.err}"))

    # sacct -P output is "<jobid>|<state>" per line, but when the command is
    # routed through ECaccess the stdout also contains the ECMWF job-filter
    # banner and ecprofile/ecepilog blocks. Only trust rows whose first column
    # is one of the job ids we actually asked about.
    expected_ids = set(slurm_job_ids)
    states: dict[str, SlurmJobState] = {}
    for line in res.stdout.splitlines():
        if not line:
            continue
        job_id, sep, raw_state = line.partition("|")
        if not sep or job_id not in expected_ids:
            continue
        # State may carry a suffix like "CANCELLED by 12345" — keep the first token
        token = raw_state.split(maxsplit=1)[0] if raw_state else ""
        if token not in _KNOWN_STATES:
            return OpError(
                err=ValueError(f"unrecognised slurm state for job {job_id!r}: {raw_state!r}")
            )
        states[job_id] = cast(SlurmJobState, token)

    missing = [jid for jid in slurm_job_ids if jid not in states]
    # TODO: happens when job is very new. Add as unknown state.
    if missing:
        return OpError(err=LookupError(f"sacct returned no record for job ids: {missing}"))

    return [SlurmJobInfo(jid, states[jid]) for jid in slurm_job_ids]


async def await_completion(
    ctx: CmdContext,
    job_ids: list[SlurmJobId],
    logger: logging.Logger,
    poll_interval_secs: int = 5,
) -> Result[list[SlurmJobInfo]]:
    """Poll sacct until all the specified jobs are in a terminal state.

    Returns the final SlurmJobInfo list (in the same order as `job_ids`) once
    every job reaches a terminal state, or the underlying OpError if a poll
    fails. The function uses asyncio.sleep, so it cooperates with other tasks
    running in the same event loop.
    """
    if not job_ids:
        return []

    # Cache jobs as they reach a terminal state so we don't re-poll them.
    # sacct is the cluster-wide accounting query — for large fan-outs this
    # avoids querying long-finished jobs every poll cycle.
    done: dict[SlurmJobId, SlurmJobInfo] = {}
    while True:
        pending_ids = [jid for jid in job_ids if jid not in done]
        if not pending_ids:
            logger.info(f"All {len(job_ids)} job(s) reached a terminal state.")
            # Preserve the input order in the returned list.
            return [done[jid] for jid in job_ids]

        infos = await get_slurm_job_states(ctx, pending_ids, logger)
        if is_err(infos):
            return infos

        for info in infos:
            if is_terminal_state(info.state):
                done[info.job_id] = info

        if len(done) < len(job_ids):
            still_active = [i for i in infos if not is_terminal_state(i.state)]
            logger.info(
                f"{len(still_active)}/{len(job_ids)} job(s) still active: "
                f"{[(i.job_id, i.state) for i in still_active]}; "
                f"sleeping {poll_interval_secs}s"
            )
            await asyncio.sleep(poll_interval_secs)


async def submit_slurm(
    job: SlurmJob, ctx: CmdContext, logger: logging.Logger
) -> Result[SlurmSubmissionResult]:
    """
    Submits a Slurm job using the provided context and logger.

    This function constructs the appropriate sbatch command based on the SlurmJob details and
    submits it using the run_cmd utility.

    Even if no stdout path is provided, this function will always construct one.
    This is necessary to fully capture the final state of the job.

    Requirements:
    If the working directory is provided, it must be absolute or start with a remote-home
    reference (`~` or `~user`). If the submission directory is provided, it must be absolute.
    If the working directory is not provided, the script path must be absolute.
    If stdout is not provided, it defaults to
    "{working_directory}/slurm_job_{job_name}_%j.out" if working_directory is
    provided, or "{script_path_parent}/slurm_job_{job_name}_%j.out".
    If stderr is not provided, it merges into the stdout file (sbatch's
    default behavior when only --output is set). To capture stderr separately,
    set `stderr` explicitly on the SlurmJob.

    It returns a SlurmSubmissionResult containing the job ID if submission is successful, or an
    OpError if there was an issue.
    """
    # TODO: the submission may fail because we do not have valid
    # credentials against the cluster. This is detected by OpError with ConnectionClosedError.
    # In this case, we should be lenient and not crash the whole flow, but
    # rather try again for a certain amount of time (up to hours for this specific case).

    # Validate path requirements.
    if job.working_directory is not None and not _is_absolute_or_remote_home_path(
        job.working_directory
    ):
        logger.error(f"working_directory must be absolute: {job.working_directory}")
        return OpError(
            err=ValueError(f"working_directory must be absolute: {job.working_directory}")
        )
    if job.submission_directory is not None and not Path(job.submission_directory).is_absolute():
        return OpError(
            err=ValueError(f"submission_directory must be absolute: {job.submission_directory}")
        )
    if (
        job.working_directory is None
        and job.script_path is not None
        and not Path(job.script_path).is_absolute()
    ):
        return OpError(
            err=ValueError(
                f"script_path must be absolute when working_directory is not set: {job.script_path}"
            )
        )

    # Resolve stdout path. Always populated so the final job state can be
    # captured even when the caller didn't ask for a specific file.
    if job.stdout is not None:
        stdout_path = Path(job.stdout)
    elif job.working_directory is not None:
        stdout_path = Path(job.working_directory) / f"slurm_job_{job.job_name}_%j.out"
    elif job.script_path is not None:
        stdout_path = Path(job.script_path).parent / f"slurm_job_{job.job_name}_%j.out"
    else:
        return OpError(
            err=ValueError(
                "Cannot determine stdout path: provide stdout, working_directory, or script_path"
            )
        )

    # Stderr default: merge into stdout (sbatch's behavior when only --output
    # is set). Only emit --error if the caller asked for a separate file.
    stderr_path = Path(job.stderr) if job.stderr is not None else stdout_path

    # Construct sbatch command. Path-valued options are passed as separate
    # shell words (`--chdir ~` rather than `--chdir=~`) so the remote shell can
    # expand leading-tilde paths before `sbatch` receives them.
    cmd_parts = [
        "sbatch",
        f"--job-name={shlex.quote(job.job_name)}",
        "--output",
        quote_path(stdout_path),
    ]
    if job.stderr is not None:
        cmd_parts.extend(["--error", quote_path(stderr_path)])

    if job.working_directory is not None:
        cmd_parts.extend(["--chdir", quote_path(job.working_directory)])

    if job.time_limit is not None:
        cmd_parts.append(f"--time={job.time_limit}")

    # Resolve --account from job.account, possibly using the context as a
    # fallback when "auto" is requested. None means "don't pass --account".
    account = _resolve_account(job.account, ctx)
    if account is not None:
        cmd_parts.append(f"--account={account}")

    if job.slurm_options:
        for option, value in job.slurm_options.items():
            cmd_parts.append(f"--{option}={value}")

    if job.script_path:
        cmd_parts.append(quote_path(job.script_path))
    elif job.command:
        # list[str] follows subprocess argv semantics: each element is one
        # token, joined into a shell line via shlex. Then shlex.quote wraps
        # the whole thing as a single argument to --wrap so embedded quotes
        # in the user's command (e.g. inline Python) survive intact.
        wrap_value = shlex.join(job.command) if isinstance(job.command, list) else job.command
        cmd_parts.append(f"--wrap={shlex.quote(wrap_value)}")
    else:
        return OpError(err=ValueError("Either script_path or command must be provided in SlurmJob"))

    full_command = " ".join(cmd_parts)
    logger.info(f"Submitting Slurm job with command: {full_command}")

    cmd = Command(command=full_command, working_directory=job.submission_directory)
    result = await run_cmd(ctx, cmd, logger)
    logger.debug(f"Slurm submission command result: {result}")
    if is_err(result):
        return result

    logger.info(
        f"Slurm job submitted successfully with output: "
        f"stdout: {result.stdout.strip()} stderr: {result.stderr.strip()}"
    )

    # Extract job id from sbatch output.
    parsed_id = _parse_job_id(result)
    if parsed_id is None:
        return OpError(err=SubmissionError(result))
    job_id: SlurmJobId = parsed_id

    # Resolve %j (and %J) in the output paths now that we know the job id, so
    # the returned paths point to the actual files Slurm will write.
    def _resolve(p: Path) -> Path:
        return Path(str(p).replace("%j", job_id).replace("%J", job_id))

    return SlurmSubmissionResult(
        job_id=job_id,
        stdout=_resolve(stdout_path),
        stderr=_resolve(stderr_path),
    )


def _resolve_account(account: str | Literal["auto"] | None, ctx: CmdContext) -> str | None:
    """Map a `SlurmJob.account` setting to the actual `--account` argument value.

    - None       -> None (no --account passed; cluster's default applies)
    - "auto"     -> the account carried on the context (if any), else None.
                    The context may not expose an account (e.g. plain SSH, local);
                    silent fallback keeps "auto" safe to set as a project default.
    - any string -> verbatim (passed straight through to sbatch).
    """
    if account is None:
        return None
    if account == "auto":
        return slurm_account(ctx)
    return account


def _parse_job_id(result: CommandResult) -> SlurmJobId | None:
    """Pull the Slurm job id from sbatch's stdout, or return None if the
    submission output is not in the expected shape (non-zero exit, empty
    stdout, or trailing token that doesn't look like an integer / array id).
    """
    if result.return_code != 0:
        return None
    tokens = result.stdout.strip().split()
    if not tokens:
        return None
    candidate = tokens[-1]
    # Slurm job ids are positive integers; array jobs append "_<index>".
    base = candidate.split("_", 1)[0]
    if not base.isdigit():
        return None
    return candidate
