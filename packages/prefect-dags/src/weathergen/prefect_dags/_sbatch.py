"""
Wrapper for prefect tasks to submit and monitor sbatch jobs on HPCs.

Implementation notes:
This wrapper is rather complex because of the following:
- handling transient errors (network failures, sacct outages, prefect crash, driver script crash)
  without losing the state
- avoid duplicate slurm submitions on retries (best effort)
- limit the number of concurrent polling calls to the HPC (1 per HPC at a time) to avoid
  overwhelming it with sacct calls.
This requires coordination between all running jobs to prevent duplicatte polling.
This itself has to account for issues like crashes during polling.

As a result:
- the submission step is cached as a separate task, so on retry we replay the cached submission
  result instead of re-submitting (which would create a new job on the HPC).
- we use prefect variables as a shared source of truth for the job status, which the monitoring
  loop updates and the polling steps read from. This way, if the polling loop crashes or
  encounters a transient error, we don't lose the state and can resume from the last known
  status on retry.
- we implement a leasing mechanism for the polling loop to ensure only one active poller per HPC
  at a time. The lease is stored in a prefect variable with an expiry time; any poller that
  finds an expired lease can attempt to acquire it by writing a new lease with a fresh expiry.
  This way, if a poller crashes while holding the lease, other pollers can detect the expired
  lease and take over the polling responsibilities without manual intervention.


"""

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID

import httpx
import prefect.runtime.flow_run
from prefect import get_client
from prefect.artifacts import acreate_markdown_artifact
from prefect.client.schemas.objects import StateType
from prefect.concurrency.asyncio import concurrency as _async_concurrency
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound
from prefect.transactions import CommitMode, transaction
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.variables import Variable
from pydantic import TypeAdapter, ValidationError

from weathergen.prefect_dags.cmd_runners import (
    CmdContext,
    Command,
    CommandRunner,
    get_command_runner,
    run_cmd,
)
from weathergen.prefect_dags.prefect import get_run_logger, task
from weathergen.prefect_dags.result import OpError, Result, is_err
from weathergen.prefect_dags.run import get_head_tail_logs
from weathergen.prefect_dags.slurm import (
    SlurmJob,
    SlurmJobId,
    SlurmJobState,
    SlurmSubmissionResult,
    get_slurm_job_states,
    is_terminal_state,
    submit_slurm,
)

# Lease window: only one monitor per HPC actively polls sacct per window.
# Others read the per-job status variables that monitor updated.
_LEASE_DURATION = timedelta(seconds=20)
# How often wait_completion_single re-checks (status variable + lease).
_POLL_INTERVAL_SECS = 5
# How long wait_completion_single tolerates a Prefect API outage (server
# restart, network blip) before giving up with an OpError, and the cap on
# the retry backoff during the outage.
_API_OUTAGE_TOLERANCE_SECS = 300
_API_OUTAGE_MAX_BACKOFF_SECS = 60

_PRFEFECT_NUM_READS = 200
# The children processes can update for 3 minutes, after which the lease expires.
# Useful for hard crashes.
_PREFECT_LEASE_DURATION_SECS = 180

_ensured_limits: set[str] = set()


async def _ensure_concurrency_limit(name: str) -> None:
    if name in _ensured_limits:
        return
    # Prefect's "upsert" is racy: it reads then creates, so two callers can
    # both see "missing" and both call create — the loser gets a 409. The
    # limit already existing is exactly the state we wanted, so swallow it.
    async with get_client() as client:
        with contextlib.suppress(ObjectAlreadyExists):
            await client.upsert_global_concurrency_limit_by_name(name=name, limit=1)
    _ensured_limits.add(name)


@dataclass
class SlurmJobResult:
    """
    The final result of a slurm job.
    """

    job_id: SlurmJobId
    status: SlurmJobState
    submission: SlurmSubmissionResult


class SlurmTerminalError(Exception):
    """
    Exception raised when a slurm job ends in a non-successful terminal state.
    """

    result: SlurmJobResult

    def __init__(self, result: SlurmJobResult):
        super().__init__(
            f"Slurm job {result.job_id} ended with non-successful state: {result.status}"
        )
        self.result = result


@task(task_run_name="sbatch-{job_name}")
def sbatch(
    ctx: CmdContext,
    *,
    job_name: str,
    stdout: str | Path | None = None,
    stderr: str | Path | None = None,
    script_path: str | Path | None = None,
    command: str | list[str] | None = None,
    working_directory: str | Path | None = None,
    submission_directory: str | Path | None = None,
    time_limit: str | None = None,
    slurm_options: dict[str, str] | None = None,
    account: str | Literal["auto"] | None = "auto",
    job: SlurmJob | None = None,
    fetch_output: bool = True,
    _accept_failure: bool = False,  # Internal, non-documented.
) -> SlurmJobResult:
    """
    Submits a slurm job and await its completion, returning the final state.

    job: if provided, the other parameters are ignored and the job is submitted as-is.


    Keyword arguments mirror `SlurmJob` — see that dataclass for the meaning
    of each field. `ctx` is the command-runner context (local / SSH / etc.).

    Throws on any issue running the job, or if the slurm job ends in a
    non-successful terminal state (e.g. "FAILED", "CANCELLED", "TIMEOUT").
    The raised exception preserves the original cause; the message includes
    the job id and final state when known.

    If you want more control over error handling or the final state of the
    job, use `sbatch_try` instead — it returns a Result.
    """
    if job is None:
        job = SlurmJob(
            job_name=job_name,
            stdout=stdout,
            stderr=stderr,
            script_path=script_path,
            command=command,
            working_directory=working_directory,
            submission_directory=submission_directory,
            time_limit=time_limit,
            slurm_options=slurm_options,
            account=account,
        )
    res = run_coro_as_sync(_sbatch_try(job, ctx, fetch_output))
    if isinstance(res, OpError):
        # Re-raise the original exception so the type and traceback are
        # preserved (e.g. FileNotFoundError on a missing script_path).
        raise res.err
    job_res: SlurmJobResult = res
    if job_res.status != "COMPLETED" and not _accept_failure:
        raise SlurmTerminalError(result=job_res)
    return job_res


def sbatch_try(
    ctx: CmdContext,
    *,
    job_name: str | None = None,
    stdout: str | Path | None = None,
    stderr: str | Path | None = None,
    script_path: str | Path | None = None,
    command: str | list[str] | None = None,
    working_directory: str | Path | None = None,
    submission_directory: str | Path | None = None,
    time_limit: str | None = None,
    slurm_options: dict[str, str] | None = None,
    account: str | Literal["auto"] | None = "auto",
    # library-specific parameters
    job: SlurmJob | None = None,
    fetch_output: bool = False,
) -> Result[SlurmJobResult]:
    """
    Same as `sbatch`, but returns a Result instead of throwing on failure.

    job: if provided, the other parameters are ignored and the job is submitted as-is.

    The signature mirrors `sbatch`. A non-successful terminal Slurm state (TIMEOUT,
    FAILED, CANCELLED, ...) becomes a successful return value here — inspect
    `result.status` to branch on it. Only exceptions raised by the
    underlying call (e.g. unreachable HPC) are wrapped as OpError.
    """
    # Delegate to `sbatch` (the cached @task) with `_accept_failure=True`
    # rather than to `_sbatch_try` directly: if we returned an OpError from
    # a Prefect task, Prefect would persist it as the task's result and
    # replay the cached failure on every rerun with the same `rerun_token`.
    if job is None:
        if job_name is None:
            return OpError(err=ValueError("Either 'job' or 'job_name' must be provided"))
        job = SlurmJob(
            job_name=job_name,
            stdout=stdout,
            stderr=stderr,
            script_path=script_path,
            command=command,
            working_directory=working_directory,
            submission_directory=submission_directory,
            time_limit=time_limit,
            slurm_options=slurm_options,
            account=account,
        )
    try:
        return sbatch(
            ctx, job=job, job_name=job.job_name, _accept_failure=True, fetch_output=fetch_output
        )

    except Exception as e:
        return OpError(err=e)


def sbatch_submit(
    job: SlurmJob,
    context: CmdContext,
) -> Result[SlurmSubmissionResult]:
    """
    Submit a slurm job without awaiting completion.

    Synchronous facade over the cached async task `_sbatch_submit_async`.
    Calling this from any flow records a cached `_sbatch_submit_async` task
    run in the graph: on rerun (with a `rerun_token`), the cached submission
    result is returned without re-launching the job. This is what `sbatch`
    relies on for idempotent recovery — a failure during polling no longer
    causes a duplicate slurm submission on retry.

    Submission failures are NOT cached: `_sbatch_submit_async` raises on
    error, Prefect marks that task run as Failed (no result persisted), and
    the next rerun will retry from scratch instead of replaying a transient
    network / auth failure. Here we catch that exception and return it as an
    OpError so the caller's Result contract is preserved.

    On purpose kept small and focused: any crash here may trigger a slurm
    job that we will not monitor.
    """
    try:
        return run_coro_as_sync(_sbatch_submit_async(job, context))
    except Exception as e:
        return OpError(err=e)


@task(task_run_name="sbatch-submit-{job.job_name}")
async def _sbatch_submit_async(
    job: SlurmJob,
    context: CmdContext,
) -> SlurmSubmissionResult:
    """Cached submission step. Called from both the sync `sbatch_submit`
    facade and the async `_sbatch_try` polling path so that on rerun a
    *successful* submission is replayed from cache instead of re-submitted.

    Returns the SlurmSubmissionResult on success; raises on failure. Raising
    (vs. returning an OpError) is deliberate: Prefect persists only the
    results of successful task runs, so a failure here is NOT cached and a
    subsequent rerun with the same rerun_token retries the submission instead
    of replaying the cached error.
    """
    logger = get_run_logger()
    submission_result = await submit_slurm(job, context, logger)
    if isinstance(submission_result, OpError):
        raise submission_result.err
    sub_res: SlurmSubmissionResult = submission_result
    logger.info(f"Submitted SLURM job with ID: {sub_res.job_id}")
    return sub_res


@dataclass
class _SlurmJobPrefectStatus:
    """
    The status of a slurm job as represented in prefect variables.
    """

    job_id: SlurmJobId
    hpc: str
    # Typed as a Literal so TypeAdapter rejects unknown state strings on read.
    # "CANCELLING" is not a slurm state: it is set by _flow_cancellation_guard
    # between committing to scancel and slurm reporting the terminal CANCELLED.
    status: SlurmJobState | Literal["CANCELLING"]
    # The flow run that created this variable (i.e. submitted or first awaited
    # the job). Preserved across sacct refreshes so cancellation of a flow run
    # only affects its own jobs. None when created outside a flow run context.
    flow_run_id: str | None = None


@dataclass
class _SlurmJobMonitoringLock:
    """
    A lock to prevent multiple concurrent monitors for the same HPC.
    """

    hpc: str
    # The time at which the lease expires in ISO format.
    lease_expires_at: str


# Pydantic adapters: validate prefect-variable values against the dataclass
# schemas at every boundary. Built once at import-time; cheap to reuse.
_STATUS_ADAPTER = TypeAdapter(_SlurmJobPrefectStatus)
_LOCK_ADAPTER = TypeAdapter(_SlurmJobMonitoringLock)


# ---------------------------------------------------------------------------
# Variable naming + helpers
# ---------------------------------------------------------------------------


async def _sbatch_try(
    job: SlurmJob,
    context: CmdContext,
    fetch_output: bool,
) -> Result[SlurmJobResult]:
    """
    Implementation of sbatch_try.
    It is separated so that the graph of tasks in prefect stays clean (no nested sbatch_try
    inside sbatch)
    """
    logger = get_run_logger()
    # Commit the submission eagerly so a downstream failure (polling crash,
    # caller's exception after sbatch returns, etc.) does NOT roll back the
    # cached submission. Without this, Prefect's transactional semantics
    # discard the child task's result when the parent fails, causing a
    # duplicate slurm submission on the next rerun.
    #
    # `_sbatch_submit_async` raises on failure (so Prefect does not cache the
    # error); convert that back into the Result contract here.
    try:
        with transaction(commit_mode=CommitMode.EAGER):
            sub_res: SlurmSubmissionResult = await _sbatch_submit_async(job, context)
    except Exception as e:
        return OpError(err=e)
    job_id = sub_res.job_id
    _runner = get_command_runner(context)
    if isinstance(_runner, Exception):
        return OpError(err=_runner)
    runner: CommandRunner = _runner
    logger.info(f"Submitted SLURM job with ID: {job_id} on hpc {runner.hpc}")
    # Insert a prefect variable so the monitoring loop can discover this job.
    # Initial "PENDING" is accurate for a freshly submitted job and will be
    # overwritten by the first lease holder that polls sacct.
    await _set_status(runner.hpc, job_id, "PENDING")
    # Surface the submission details in the Prefect UI as a markdown artifact.
    # Artifacts render directly on the task run page.
    artifact_key = (
        f"slurm-submission-{_artifact_key_part(job.job_name)}-{_artifact_key_part(job_id)}"
    )
    artifact_description = f"slurm job {job_id} on {runner.hpc}"
    submission_md = (
        f"## SLURM submission\n\n"
        f"- **Job ID:** `{job_id}`\n"
        f"- **HPC:** `{runner.hpc}`\n"
        f"- **Job name:** `{job.job_name}`\n"
        f"- **stdout:** `{sub_res.stdout}`\n"
        f"- **stderr:** `{sub_res.stderr}`\n"
    )
    _ = await acreate_markdown_artifact(
        key=artifact_key,
        markdown=submission_md,
        description=artifact_description,
    )

    # Now await completion of the job. The status variable is cleaned up by
    # wait_completion_single once the job reaches a terminal state.
    state = await wait_completion_single(logger, context, job_id, runner)
    if isinstance(state, OpError):
        return state

    if fetch_output:
        # Fetch head and tail of stdout and stderr for debugging purposes.
        output = get_head_tail_logs(
            ctx=context,
            stdout_path=str(sub_res.stdout),
            stderr_path=str(sub_res.stderr) if sub_res.stderr else None,
        )
        logger.info(f"Fetched head/tail logs for job {job_id} on hpc {runner.hpc}: {output}")
        # Re-create the artifact with the same key: Prefect supersedes the previous
        # version, so the task-run page shows the submission info plus the captured
        # output as a single artifact.
        _ = await acreate_markdown_artifact(
            key=artifact_key,
            markdown=submission_md + "\n" + _format_output_section(output),
            description=artifact_description,
        )

    return SlurmJobResult(
        job_id=job_id,
        status=state,
        submission=sub_res,
    )


def _format_output_section(output: dict[str, str]) -> str:
    # Use 4-backtick fences so log content containing ``` doesn't break the block.
    def section(title: str, body: str | None) -> str:
        if body is None:
            return f"### {title}\n\n_(not available)_\n\n"
        if body == "":
            return f"### {title}\n\n_(empty)_\n\n"
        return f"### {title}\n\n````\n{body}\n````\n\n"

    return (
        "## SLURM job output\n\n"
        + section("stdout (head)", output.get("head_out"))
        + section("stdout (tail)", output.get("tail_out"))
        + section("stderr (head)", output.get("head_err"))
        + section("stderr (tail)", output.get("tail_err"))
    )


def _artifact_key_part(s: str) -> str:
    """Sanitize a string for use in a Prefect artifact key.

    Prefect requires keys to match `^[a-z0-9-]+$`. We lowercase, replace any
    run of disallowed characters with a single dash, and strip leading /
    trailing dashes. Empty results fall back to "x" so the surrounding
    f-string never produces consecutive dashes.
    """
    cleaned = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return cleaned or "x"


def _status_var_name(hpc: str, job_id: SlurmJobId) -> str:
    return f"sbatch_{hpc}_{job_id}"


def _lease_var_name(hpc: str) -> str:
    return f"_sbatch_monitoring_{hpc}"


def _now_utc() -> datetime:
    return datetime.now(UTC)


async def _set_status(
    hpc: str, job_id: SlurmJobId, status: SlurmJobState | Literal["CANCELLING"]
) -> None:
    # Preserve the flow run that originally created this variable: the
    # lease-holding monitor refreshes variables for jobs belonging to *other*
    # flow runs and must not re-stamp them as its own (otherwise cancelling
    # one flow run could scancel another's jobs).
    existing = await _read_status_payload(hpc, job_id)
    flow_run_id = (
        existing.flow_run_id
        if existing is not None and existing.flow_run_id
        else prefect.runtime.flow_run.id or None
    )
    payload = _SlurmJobPrefectStatus(job_id=job_id, hpc=hpc, status=status, flow_run_id=flow_run_id)
    tags = [f"hpc:{hpc}", "status:active"]
    # Tag with the flow currently running this task, to ease tracing a status
    # variable back to its flow in the UI. May be absent (e.g. unit tests).
    flow_name: str = prefect.runtime.flow_run.flow_name
    if flow_name:
        tags.append(f"flow:{flow_name}")
    if flow_run_id:
        tags.append(f"flow_run:{flow_run_id}")
    _ = await Variable.aset(
        name=_status_var_name(hpc, job_id),
        value=_STATUS_ADAPTER.dump_python(payload),
        tags=tags,
        overwrite=True,
    )


async def _read_status_payload(hpc: str, job_id: SlurmJobId) -> _SlurmJobPrefectStatus | None:
    raw = await Variable.aget(name=_status_var_name(hpc, job_id))
    if raw is None:
        return None
    try:
        return _STATUS_ADAPTER.validate_python(raw)
    except ValidationError:
        # Stale or malformed variable from an older schema; treat as missing.
        return None


async def _read_status(
    hpc: str, job_id: SlurmJobId
) -> SlurmJobState | Literal["CANCELLING"] | None:
    parsed = await _read_status_payload(hpc, job_id)
    return parsed.status if parsed is not None else None


async def _delete_status(hpc: str, job_id: SlurmJobId) -> None:
    _ = await Variable.aunset(name=_status_var_name(hpc, job_id))


async def _list_status_payloads(hpc: str, logger: logging.Logger) -> list[_SlurmJobPrefectStatus]:
    # Prefect's client doesn't expose server-side filtering on variables, so
    # we read a bounded batch and filter by tags client-side. Fine for the
    # typical "tens of active jobs per HPC" scale.
    async with get_client() as client:
        results = await client.read_variables(limit=_PRFEFECT_NUM_READS)
    needed = {f"hpc:{hpc}", "status:active"}
    out: list[_SlurmJobPrefectStatus] = []
    for v in results:
        if not needed.issubset(set(v.tags)):
            continue
        try:
            parsed = _STATUS_ADAPTER.validate_python(v.value)
        except ValidationError as e:
            logger.warning(f"Malformed status variable: {v.name}: {v.value}: {v.tags} : {e}")
            # Skip malformed values rather than failing the whole refresh.
            continue
        out.append(parsed)
    return out


async def _list_running_job_ids(hpc: str, logger: logging.Logger) -> list[SlurmJobId]:
    payloads = await _list_status_payloads(hpc, logger)
    # Jobs in CANCELLING are being folded by _flow_cancellation_guard: skip
    # them so the sacct refresh does not overwrite the marker (sacct may
    # still briefly report them as RUNNING after scancel).
    return [p.job_id for p in payloads if p.status != "CANCELLING"]


async def _lease_expired(hpc: str) -> bool:
    """Return True if the lease is missing, malformed, or past expiry."""
    raw = await Variable.aget(name=_lease_var_name(hpc))
    if raw is None:
        return True
    try:
        lock = _LOCK_ADAPTER.validate_python(raw)
    except ValidationError:
        return True  # malformed -> treat as expired
    try:
        expires = datetime.fromisoformat(lock.lease_expires_at)
    except ValueError:
        return True
    return expires <= _now_utc()


async def _write_lease(hpc: str) -> None:
    """Write a fresh lease (overwrites any existing one)."""
    lock = _SlurmJobMonitoringLock(
        hpc=hpc,
        lease_expires_at=(_now_utc() + _LEASE_DURATION).isoformat(),
    )
    _ = await Variable.aset(
        name=_lease_var_name(hpc),
        value=_LOCK_ADAPTER.dump_python(lock),
        overwrite=True,
    )


async def wait_completion_single(
    logger: logging.Logger,
    ctx: CmdContext,
    job_id: SlurmJobId,
    runner: CommandRunner,
) -> Result[SlurmJobState]:
    """Poll the per-job status variable until the job reaches a terminal state.

    Each iteration:
      1. Try to refresh sacct (only succeeds if we win the HPC lease).
      2. Read this job's status variable.

    If the status variable is missing (e.g. the job was submitted outside of
    `sbatch`, such as by an external launcher script), it is created with a
    PENDING status so the monitor loop picks the job up. The variable is
    deleted once the job reaches a terminal state.

    Transient Prefect API outages (server restart, network blip, laptop
    sleep) are tolerated: polling retries with backoff for up to
    `_API_OUTAGE_TOLERANCE_SECS` before giving up with an OpError. The slurm
    jobs themselves are unaffected by such outages.
    """
    outage_started: datetime | None = None
    backoff = _POLL_INTERVAL_SECS
    while True:
        try:
            result = await _poll_completion_once(logger, ctx, job_id, runner)
        except (httpx.HTTPError, OSError) as e:
            # The Prefect API is unreachable. This is not a problem with the
            # slurm job: keep polling with backoff for a bounded window.
            now = _now_utc()
            outage_started = outage_started or now
            elapsed = (now - outage_started).total_seconds()
            if elapsed > _API_OUTAGE_TOLERANCE_SECS:
                logger.error(
                    f"Prefect API unreachable for {elapsed:.0f}s while waiting for "
                    f"slurm job {job_id} on hpc {runner.hpc}; giving up: {e}"
                )
                return OpError(err=e)
            logger.warning(
                f"Prefect API unreachable ({e}); retrying in {backoff}s "
                f"(outage tolerance: {elapsed:.0f}/{_API_OUTAGE_TOLERANCE_SECS}s)"
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _API_OUTAGE_MAX_BACKOFF_SECS)
            continue
        outage_started = None
        backoff = _POLL_INTERVAL_SECS
        if result is not None:
            return result
        await asyncio.sleep(_POLL_INTERVAL_SECS)


async def _poll_completion_once(
    logger: logging.Logger,
    ctx: CmdContext,
    job_id: SlurmJobId,
    runner: CommandRunner,
) -> Result[SlurmJobState] | None:
    """One iteration of the wait_completion_single poll loop.

    Returns the final Result when the job is done (or the guard/scancel
    failed), or None to keep polling. Prefect API connection failures are
    raised and handled by the caller's outage tolerance.
    """
    guard = await _flow_cancellation_guard(logger, ctx, runner)
    if is_err(guard):
        return guard
    if guard.cancelled:
        # The user asked to cancel this flow run; its jobs (including this
        # one) have been committed to scancel. Fold immediately rather
        # than waiting for sacct to report the terminal state.
        logger.info(f"SLURM job {job_id} cancelled with its flow run.")
        await _delete_status(runner.hpc, job_id)
        return "CANCELLED"
    await _try_update_status(logger, ctx, runner)
    status = await _read_status(runner.hpc, job_id)
    if status is None:
        # Variable missing: either it should not happen (sbatch path) or
        # the job was submitted externally and is awaited for the first time.
        # Insert the variable with a "PENDING" status so the monitor loop
        # can update the status.
        logger.warning(
            f"Status variable for job {job_id} on hpc {runner.hpc} is missing. "
            f"Reinserting with PENDING status."
        )
        await _set_status(runner.hpc, job_id, "PENDING")
        return None
    if status == "CANCELLING":
        # Marked by the cancellation guard (from a concurrent task of this
        # flow run). The guard at the top of the next iteration folds.
        return None
    if is_terminal_state(status):
        logger.info(f"SLURM job {job_id} reached terminal state: {status}")
        await _delete_status(runner.hpc, job_id)
        return status
    return None


@dataclass
class CancelledSlurmJobs:
    # The list of slurm jobs that have been committed to cancellation with scancel.
    cancelled: list[SlurmJobId]


async def _flow_cancellation_guard(
    logger: logging.Logger, ctx: CmdContext, runner: CommandRunner
) -> Result[CancelledSlurmJobs]:
    """
    Check whether the user asked to cancel the current flow run, and if so,
    fold all the slurm jobs associated with it.

    Local (non-deployment) flow runs have no worker to enforce cancellation,
    so the UI cancel button is greyed out. The cooperative signals are:
    - the flow run was forced into Cancelling/Cancelled state by the user
      (UI kebab menu "Change state", or `prefect flow-run cancel <id>`);
    - the flow run was deleted in the UI.

    When a signal is detected:
    - mark preemptively as CANCELLING (with _set_status) all the slurm jobs
      associated with that flow run, so the sacct monitor stops refreshing them;
    - call scancel on all those jobs. If scancel fails, return the error.

    Returns the (possibly empty) list of jobs committed to cancellation.
    Transient API failures while checking the state are logged and treated as
    "no signal": a network blip must not cancel real work.
    """
    no_signal = CancelledSlurmJobs(cancelled=[])
    flow_run_id: UUID | None = prefect.runtime.flow_run.id
    if not flow_run_id:
        # Not running inside a flow run (e.g. unit tests): nothing to guard.
        return no_signal

    # 1. Detect the cancellation signal.
    try:
        async with get_client() as client:
            flow_run = await client.read_flow_run(flow_run_id)
        if flow_run.state_type not in (StateType.CANCELLING, StateType.CANCELLED):
            return no_signal
        reason = f"flow run {flow_run_id} was put in state {flow_run.state_type}"
    except ObjectNotFound:
        reason = f"flow run {flow_run_id} was deleted"
    except Exception as e:
        logger.warning(f"Could not check the cancellation state of flow run {flow_run_id}: {e}")
        return no_signal

    logger.warning(f"Cancellation requested: {reason}. Cancelling its slurm jobs.")

    # 2. Find all the jobs of this flow run on this HPC and mark them as
    # CANCELLING so the sacct refresh loop leaves them alone.
    payloads = await _list_status_payloads(runner.hpc, logger)
    job_ids = [p.job_id for p in payloads if p.flow_run_id == flow_run_id]
    for jid in job_ids:
        await _set_status(runner.hpc, jid, "CANCELLING")

    # 3. Cancel them on the HPC. scancel is idempotent on jobs that are
    # already terminal, so concurrent guards racing here are harmless.
    if job_ids:
        res = await run_cmd(ctx, Command(command=["scancel", *job_ids]), logger)
        if is_err(res):
            return res
        if res.return_code != 0:
            return OpError(
                err=RuntimeError(f"scancel failed for jobs {job_ids} on hpc {runner.hpc}"),
                stdout=res.stdout,
                stderr=res.stderr,
                return_code=res.return_code,
            )
        logger.info(f"scancel issued for jobs {job_ids} on hpc {runner.hpc}")
    return CancelledSlurmJobs(cancelled=job_ids)


async def _try_update_status(
    logger: logging.Logger,
    ctx: CmdContext,
    runner: CommandRunner,
) -> None:
    """If the HPC's lease has expired, refresh sacct for all running jobs.

    Uses a double-check-inside-a-semaphore pattern:
      1. Cheap check outside the semaphore — fast path when the lease is
         fresh, so the vast majority of polls skip the lock entirely.
      2. Acquire a per-HPC concurrency slot (we upsert the global limit
         with a single slot on first use, giving us a real mutex). Tasks
         racing here queue briefly until the current sacct call finishes.
      3. Re-check the lease inside the lock — the holder we just waited on
         may have already refreshed it. Skip if so.
      4. Write a fresh lease, then do the actual sacct refresh.
    """
    # 1. Fast path.
    if not await _lease_expired(runner.hpc):
        # The lease has not expired, another task may be refreshing right now.
        return

    # 2. Per-HPC critical section.
    limit_name = f"sacct-refresh-{runner.hpc}"
    logger.info(f"Lease expired for hpc {runner.hpc}, attempting to acquire lock for sacct refresh")
    await _ensure_concurrency_limit(limit_name)
    logger.info(f"Ensured concurrency limit {limit_name} for sacct refresh")
    async with _async_concurrency(
        limit_name, lease_duration=_PREFECT_LEASE_DURATION_SECS, strict=True
    ):
        # 3. Double check — another task may have refreshed while we waited.
        logger.info(f"Acquired lock for hpc {runner.hpc}, re-checking lease")
        if not await _lease_expired(runner.hpc):
            return
        logger.info(f"Lease still expired for hpc {runner.hpc} after acquiring lock, refreshing")
        # 4. We hold both lock and an expired lease. Write the new lease
        # first so any task currently in step 1 sees fresh state and bails.
        await _write_lease(runner.hpc)

        logger.info(f"Acquired lease for hpc {runner.hpc} — refreshing SLURM job states")

    # The lease has been renewed.
    # Do the refresh outside the critical section.
    logger.info(f"Refreshing SLURM job states for hpc {runner.hpc}")

    job_ids = await _list_running_job_ids(runner.hpc, logger)
    if not job_ids:
        return
    logger.info(f"Found {len(job_ids)} running jobs for hpc {runner.hpc}: {job_ids}")
    states_res = await get_slurm_job_states(ctx, job_ids, logger)
    if isinstance(states_res, OpError):
        # Don't propagate: this is opportunistic refresh. The next
        # monitor will retry. Surface in the log so the operator can
        # act on chronic failures (e.g. sacct outage, network issue).
        logger.warning(f"Failed to refresh slurm states for hpc={runner.hpc}: {states_res.err}")
        return
    for info in states_res:
        await _set_status(runner.hpc, info.job_id, info.state)
    logger.info(f"Refreshed {len(states_res)} states for hpc {runner.hpc}")
