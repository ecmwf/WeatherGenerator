"""
Wrapper for prefect tasks to submit and monitor sbatch jobs on HPCs.

"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from prefect import get_client
from prefect.artifacts import acreate_markdown_artifact
from prefect.concurrency.asyncio import concurrency as _async_concurrency
from prefect.transactions import CommitMode, transaction
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.variables import Variable
from pydantic import TypeAdapter, ValidationError

from weathergen.prefect_dags.cmd_runners import CmdContext, CommandRunner, get_command_runner
from weathergen.prefect_dags.prefect_logging import get_run_logger
from weathergen.prefect_dags.prefect_wrapper import task
from weathergen.prefect_dags.result import OpError, Result, is_err
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
# How often _wait_completion_single re-checks (status variable + lease).
_POLL_INTERVAL_SECS = 5

_PRFEFECT_NUM_READS = 200

_ensured_limits: set[str] = set()


async def _ensure_concurrency_limit(name: str) -> None:
    if name in _ensured_limits:
        return
    async with get_client() as client:
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
) -> SlurmJobResult:
    """
    Submits a slurm job and await its completion, returning the final state.

    Keyword arguments mirror `SlurmJob` — see that dataclass for the meaning
    of each field. `ctx` is the command-runner context (local / SSH / etc.).

    Throws on any issue running the job, or if the slurm job ends in a
    non-successful terminal state (e.g. "FAILED", "CANCELLED", "TIMEOUT").
    The raised exception preserves the original cause; the message includes
    the job id and final state when known.

    If you want more control over error handling or the final state of the
    job, use `sbatch_try` instead — it returns a Result.
    """
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
    res = run_coro_as_sync(_sbatch_try(job, ctx))
    if isinstance(res, OpError):
        # Re-raise the original exception so the type and traceback are
        # preserved (e.g. FileNotFoundError on a missing script_path).
        raise res.err
    job_res: SlurmJobResult = res
    if job_res.status != "COMPLETED":
        raise RuntimeError(
            f"Slurm job {job_res.job_id} ended with non-successful state: {job_res.status}"
        )
    return job_res


def sbatch_try(
    job: SlurmJob,
    context: CmdContext,
) -> Result[SlurmJobResult]:
    """
    Same as `sbatch`, but returns a Result instead of throwing on failure.
    """
    # Delegates to `sbatch` (the cached @task) rather than calling `_sbatch_try`
    # directly: if we returned an OpError from a Prefect task here, Prefect
    # would persist it as the task's result and replay the cached failure on
    # every rerun with the same `rerun_token`.
    
    try:
        return sbatch(
            context,
            job_name=job.job_name,
            stdout=job.stdout,
            stderr=job.stderr,
            script_path=job.script_path,
            command=job.command,
            working_directory=job.working_directory,
            submission_directory=job.submission_directory,
            time_limit=job.time_limit,
            slurm_options=job.slurm_options,
            account=job.account,
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
    status: SlurmJobState


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
) -> Result[SlurmJobResult]:
    """
    Implementation of sbatch_try.
    It is separated so that the graph of tasks in prefect stays clean (no nested sbatch_try inside sbatch)
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
    await acreate_markdown_artifact(
        key=f"slurm-submission-{_artifact_key_part(job.job_name)}-{_artifact_key_part(job_id)}",
        markdown=(
            f"## SLURM submission\n\n"
            f"- **Job ID:** `{job_id}`\n"
            f"- **HPC:** `{runner.hpc}`\n"
            f"- **Job name:** `{job.job_name}`\n"
            f"- **stdout:** `{sub_res.stdout}`\n"
            f"- **stderr:** `{sub_res.stderr}`\n"
        ),
        description=f"slurm job {job_id} on {runner.hpc}",
    )

    # Now await completion of the job.
    state = await _wait_completion_single(logger, context, job_id, runner)
    # Job is done. Cleane up the variable.
    await _delete_status(runner.hpc, job_id)
    if isinstance(state, OpError):
        return state

    return SlurmJobResult(
        job_id=job_id,
        status=state,
        submission=sub_res,
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


async def _set_status(hpc: str, job_id: SlurmJobId, status: SlurmJobState) -> None:
    payload = _SlurmJobPrefectStatus(job_id=job_id, hpc=hpc, status=status)
    await Variable.aset(
        name=_status_var_name(hpc, job_id),
        value=_STATUS_ADAPTER.dump_python(payload),
        tags=[f"hpc:{hpc}", "status:running"],
        overwrite=True,
    )


async def _read_status(hpc: str, job_id: SlurmJobId) -> SlurmJobState | None:
    raw = await Variable.aget(name=_status_var_name(hpc, job_id))
    if raw is None:
        return None
    try:
        parsed = _STATUS_ADAPTER.validate_python(raw)
    except ValidationError:
        # Stale or malformed variable from an older schema; treat as missing.
        return None
    return parsed.status


async def _delete_status(hpc: str, job_id: SlurmJobId) -> None:
    await Variable.aunset(name=_status_var_name(hpc, job_id))


async def _list_running_job_ids(hpc: str, logger: logging.Logger) -> list[SlurmJobId]:
    # Prefect's client doesn't expose server-side filtering on variables, so
    # we read a bounded batch and filter by tags client-side. Fine for the
    # typical "tens of active jobs per HPC" scale.
    async with get_client() as client:
        results = await client.read_variables(limit=_PRFEFECT_NUM_READS)
    needed = {f"hpc:{hpc}", "status:running"}
    out: list[SlurmJobId] = []
    for v in results:
        if not needed.issubset(set(v.tags)):
            continue
        try:
            parsed = _STATUS_ADAPTER.validate_python(v.value)
        except ValidationError as e:
            logger.warning(f"Malformed status variable: {v.name}: {v.value}: {v.tags} : {e}")
            # Skip malformed values rather than failing the whole refresh.
            continue
        out.append(parsed.job_id)
    return out


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
    await Variable.aset(
        name=_lease_var_name(hpc),
        value=_LOCK_ADAPTER.dump_python(lock),
        overwrite=True,
    )


async def _wait_completion_single(
    logger: logging.Logger,
    ctx: CmdContext,
    job_id: SlurmJobId,
    runner: CommandRunner,
) -> Result[SlurmJobState]:
    """Poll the per-job status variable until it disappears (terminal signal).

    Each iteration:
      1. Try to refresh sacct (only succeeds if we win the HPC lease).
      2. Read this job's status variable.

    """
    while True:
        await _try_update_status(logger, ctx, runner)
        status = await _read_status(runner.hpc, job_id)
        if status is None:
            # Variable missing. It should not happen.
            # Reinsert the variable with a "PENDING" status so the monitor loop can update the status.
            logger.warning(
                f"Status variable for job {job_id} on hpc {runner.hpc} is missing. Reinserting with PENDING status."
            )
            await _set_status(runner.hpc, job_id, "PENDING")
        elif status is not None and is_err(status):
            logger.info(f"SLURM job {job_id} reached terminal error state: {status}")
            return status
        elif status is not None and is_terminal_state(status):
            logger.info(f"SLURM job {job_id} reached terminal state: {status}")
            return status
        await asyncio.sleep(_POLL_INTERVAL_SECS)


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
        return

    # 2. Per-HPC critical section.
    limit_name = f"sacct-refresh-{runner.hpc}"
    await _ensure_concurrency_limit(limit_name)
    async with _async_concurrency(limit_name):
        # 3. Double check — another task may have refreshed while we waited.
        if not await _lease_expired(runner.hpc):
            return

        # 4. We hold both lock and an expired lease. Write the new lease
        # first so any task currently in step 1 sees fresh state and bails.
        await _write_lease(runner.hpc)

        logger.info(f"Acquired lease for hpc {runner.hpc} — refreshing SLURM job states")

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
