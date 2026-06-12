"""
Command runner for the CSCS FirecREST v2 API.

FirecREST has no synchronous "exec" endpoint — every command must go through
Slurm. This runner wraps the requested command in a single-task batch script,
submits it via pyfirecrest, blocks on `wait_for_job` until the job completes,
then reads stdout / stderr / exit code from files written into a scratch
directory on the cluster.

# Usage instructions

- Obtain an API token from https://docs.cscs.ch/access/firecrest/
- `scratch_dir` is normally left unset: it's discovered automatically from
  the cluster's `defaultWorkDir` filesystem combined with the authenticated
  username (e.g. `/capstor/scratch/cscs/<username>`).
"""

import hashlib
import shlex
import uuid
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Literal

import firecrest as f7t

from weathergen.prefect_dags.cmd_runners._types import (
    Command,
    CommandResult,
    CommandRunner,
    quote_path,
)
from weathergen.prefect_dags.result import OpError, Result

type CscsHpc = Literal["santis", "alps"]


@dataclass
class CscsFirecrestContext:
    """
    Connection info for CSCS FirecREST v2.

    `hpc` is the FirecREST `system_name` (e.g. "santis", "alps").

    Authentication — provide one of the following, in priority order:
      1. `consumer_key` (or `consumer_key_path`) + `consumer_secret` (or
         `consumer_secret_path`): OAuth2 client credentials from the CSCS
         developer portal. The client mints short-lived JWT tokens via
         `token_uri` and refreshes them automatically. Prefer the `_path`
         variants so secrets aren't checked into source. **Preferred.**
      2. `api_token` or `api_token_path`: a pre-issued bearer token (string
         or file). Kept as a fallback for environments where only a static
         token is available.

    `token_uri` is the OIDC token endpoint used with consumer credentials;
        defaults to the CSCS firecrest-clients realm.
    `firecrest_url` is the FirecREST v2 base URL. When None (the default),
        it's derived from `hpc`: santis → `/cw/`, alps → `/hpc/`. Override
        for ML-platform machines (`/ml/`) or non-prod endpoints.
    `scratch_dir`: when None (the default), the runner asks FirecREST for
        the cluster's `defaultWorkDir` filesystem and the authenticated
        username, and stages files at `<defaultWorkDir>/<username>`. Set
        explicitly to override.
    `partition` / `account` are optional Slurm directives; some CSCS systems
        require an account.
    `wall_time` is the Slurm `--time` value (HH:MM:SS) for the wrapper job.
        Keep it small — short commands shouldn't sit in long queues.
    `wait_timeout` is the maximum seconds to wait for completion; past that
        the job is cancelled by pyfirecrest.
    """

    hpc: CscsHpc
    # Preferred: OAuth2 client credentials. The token gateway issues a fresh
    # short-lived bearer per request, refreshed automatically by pyfirecrest.
    # Pass the literal value, or a file path to read it from — the `_path`
    # variants exist so secrets don't have to be embedded in source.
    consumer_key_path: str | Path | None = None
    consumer_secret_path: str | Path | None = None
    # Alternative: provide credentials as strings.
    # WARNING: embedding secrets in source is STRONGLY DISCOURAGED.
    # If these credentials are committed to source, they will likely be leaked
    # and should be considered compromised. Rotate immediately if that happens.
    consumer_key: str | None = None
    consumer_secret: str | None = None
    token_uri: str = (
        "https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"
    )
    # Fallback: a pre-issued static bearer token (string or file). Kept for
    # environments where only an API token is available, but consumer
    # credentials are the supported path on CSCS today.
    api_token: str | None = None
    api_token_path: str | Path | None = None
    # The rest is mostly automatic.
    firecrest_url: str | None = None
    scratch_dir: str | None = None
    partition: str | None = None
    account: str | None = None
    wall_time: str = "00:10:00"
    wait_timeout: float = 600.0


class _BearerTokenAuth:
    """
    Adapter so a pre-issued bearer token satisfies pyfirecrest's authorization
    protocol (any object with `get_access_token() -> str`). `ClientCredentialsAuth`
    drives the OAuth2 flow; when the caller already has a token, this avoids
    re-issuing one on every request.
    """

    def __init__(self, token: str):
        self._token = token

    def get_access_token(self) -> str:
        return self._token


class CscsFirecrestCommandRunner(CommandRunner):
    """
    CommandRunner that executes commands on CSCS clusters via the FirecREST v2 API.

    Each `run()` submits a single-task Slurm job wrapping the command, blocks
    on `wait_for_job`, then fetches output via `view()` on the cluster.
    """

    name = "cscs_firecrest"
    _ctx: CscsFirecrestContext
    _client: f7t.v2.Firecrest
    # sha256 of the *_path credential files at the time _client was built.
    # Long-running runners outlive credential rotation (CSCS API tokens are
    # typically refreshed daily); re-hash on every run() and rebuild the
    # client whenever the bytes on disk change.
    _credential_signature: tuple[str | None, ...]
    # Memoized scratch directory. Discovery hits two REST endpoints
    # (`systems` + `userinfo`) and the answer never changes for a given
    # context, so resolve once and cache.
    _resolved_scratch_dir: str | None

    def __init__(self, context: CscsFirecrestContext):
        self._ctx = context
        self.hpc = context.hpc
        self._credential_signature = _credential_signature(context)
        self._client = _build_client(context)
        self._resolved_scratch_dir = context.scratch_dir

    def run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        self._refresh_client_if_credentials_changed(logger)
        try:
            return self._run(cmd, logger)
        except Exception as e:
            # Per the CommandRunner contract: errors are returned, not raised.
            logger.error(f"FirecREST command failed: {e}")
            return OpError(err=e)

    def _refresh_client_if_credentials_changed(self, logger: Logger) -> None:
        sig = _credential_signature(self._ctx)
        if sig == self._credential_signature:
            return
        logger.info("Credential file content changed; rebuilding FirecREST client")
        self._credential_signature = sig
        self._client = _build_client(self._ctx)

    def _run(self, cmd: Command, logger: Logger) -> Result[CommandResult]:
        ctx = self._ctx
        scratch_dir = self._get_scratch_dir(logger)
        logger.info(f"Scratch dir: {scratch_dir}")

        run_id = uuid.uuid4().hex[:12]
        stdout_path = f"{scratch_dir}/firecrest-{run_id}.out"
        stderr_path = f"{scratch_dir}/firecrest-{run_id}.err"
        rc_path = f"{scratch_dir}/firecrest-{run_id}.rc"
        logger.info(f"fc run: {run_id} rc_path: {rc_path}")

        script = _build_sbatch_script(
            cmd=cmd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            rc_path=rc_path,
            partition=ctx.partition,
            wall_time=ctx.wall_time,
        )

        # FirecREST's submit `working_dir` is just where Slurm runs the script
        # from — the in-script `cd` already handles the caller's intent, but
        # Slurm still wants *some* directory it can chdir to.
        working_dir = str(cmd.working_directory) if cmd.working_directory else scratch_dir

        logger.info(
            f"Submitting Slurm wrapper job to {ctx.hpc} via FirecREST: working dir: {working_dir}"
        )
        submit_resp = self._client.submit(
            system_name=ctx.hpc,
            working_dir=working_dir,
            script_str=script,
            env_vars=cmd.env_vars,
            account=ctx.account,
        )

        job_id = _extract_job_id(submit_resp)
        logger.info(f"Submit response: {submit_resp}, extracted job_id: {job_id}")
        if not job_id:
            return OpError(err=RuntimeError(f"Unexpected submit response: {submit_resp}"))

        logger.info(f"Submitted job {job_id}; waiting up to {ctx.wait_timeout}s for completion")
        self._client.wait_for_job(
            system_name=ctx.hpc,
            job_id=job_id,
            timeout=ctx.wait_timeout,
        )

        # Read output back. view() raises if the file is missing — which can
        # happen if the command produced no output, or never ran. Treat
        # missing as empty / -1 rather than propagating.
        stdout = _safe_view(self._client, ctx.hpc, stdout_path, logger)
        stderr = _safe_view(self._client, ctx.hpc, stderr_path, logger)
        rc_str = _safe_view(self._client, ctx.hpc, rc_path, logger).strip()
        try:
            return_code = int(rc_str) if rc_str else -1
        except ValueError:
            return_code = -1

        logger.info(f"Job {job_id} finished with return code {return_code}")
        return CommandResult(stdout=stdout, stderr=stderr, return_code=return_code)

    def _get_scratch_dir(self, logger: Logger) -> str:
        if self._resolved_scratch_dir is not None:
            return self._resolved_scratch_dir
        logger.info(f"Resolving scratch_dir for {self._ctx.hpc} via FirecREST")

        base = _find_default_work_dir(self._client, self._ctx.hpc)
        username = _find_username(self._client, self._ctx.hpc)
        # CSCS scratch is laid out as <defaultWorkDir>/<username> — see
        # the example at https://docs.cscs.ch/access/firecrest/.
        resolved = f"{base.rstrip('/')}/{username}"
        logger.info(f"Resolved scratch_dir for {self._ctx.hpc}: {resolved}")
        self._resolved_scratch_dir = resolved
        return resolved


def _default_firecrest_url(hpc: CscsHpc) -> str:
    # CSCS routes by platform path (https://docs.cscs.ch/access/firecrest/):
    # santis → C&W platform (/cw/), alps → HPC platform (/hpc/).
    # Daint/Eiger and other Alps machines all live behind /hpc/.
    if hpc == "santis":
        return "https://api.svc.cscs.ch/cw/firecrest/v2"
    platform = {"santis": "cw", "alps": "hpc"}[hpc]
    return f"https://api.cscs.ch/{platform}/firecrest/v2"


def _build_client(ctx: CscsFirecrestContext) -> f7t.v2.Firecrest:
    return f7t.v2.Firecrest(
        firecrest_url=ctx.firecrest_url or _default_firecrest_url(ctx.hpc),
        authorization=_build_authorization(ctx),
    )


def _credential_signature(ctx: CscsFirecrestContext) -> tuple[str | None, ...]:
    """sha256 of each *_path credential file. None when the path is unset
    or the file is missing — a missing file is its own distinct state, so
    appearing later still triggers a rebuild."""
    paths = (ctx.consumer_key_path, ctx.consumer_secret_path, ctx.api_token_path)
    return tuple(_hash_file(p) for p in paths)


def _hash_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _build_authorization(ctx: CscsFirecrestContext) -> object:
    # Priority 1: OAuth2 client credentials. pyfirecrest's
    # `ClientCredentialsAuth` handles minting and refreshing the JWT.
    key = _resolve_secret("consumer_key", ctx.consumer_key, ctx.consumer_key_path)
    secret = _resolve_secret("consumer_secret", ctx.consumer_secret, ctx.consumer_secret_path)
    if key and secret:
        return f7t.ClientCredentialsAuth(key, secret, ctx.token_uri)
    if key or secret:
        raise ValueError(
            "Partial OAuth2 credentials: provide both consumer_key "
            "(or consumer_key_path) and consumer_secret (or consumer_secret_path)."
        )
    # Priority 2: a pre-issued bearer token (literal or read from file).
    token = _resolve_secret("api_token", ctx.api_token, ctx.api_token_path)
    if token:
        return _BearerTokenAuth(token)
    raise ValueError(
        "No FirecREST credentials provided: set either "
        "(consumer_key + consumer_secret) or (api_token / api_token_path)."
    )


def _resolve_secret(name: str, value: str | None, path: str | Path | None) -> str | None:
    if value and path:
        raise ValueError(f"Provide exactly one of {name} or {name}_path, not both.")
    if value:
        return value
    if path:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{name}_path points to {resolved} (from {path!r}), but no such file exists."
            )
        return resolved.read_text().strip()
    return None


def _find_default_work_dir(client: f7t.v2.Firecrest, hpc: str) -> str:
    for sys_info in client.systems():
        if sys_info.get("name") != hpc:
            continue
        for fs in sys_info.get("fileSystems", []):
            if fs.get("defaultWorkDir"):
                path = fs.get("path")
                if isinstance(path, str) and path:
                    return path
        raise RuntimeError(f"No filesystem with defaultWorkDir=true on {hpc}")
    raise RuntimeError(f"System {hpc!r} not found in FirecREST systems list")


def _find_username(client: f7t.v2.Firecrest, hpc: str) -> str:
    info = client.userinfo(system_name=hpc)
    # The endpoint returns Linux-id-style info. Different FirecREST versions
    # have used slightly different shapes; try the documented one first, then
    # the obvious fallbacks before giving up.
    for path in (("user", "name"), ("username",), ("name",)):
        cur: object = info
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, str) and cur:
            return cur
    raise RuntimeError(f"Could not find a username in userinfo response: {info!r}")


def _build_sbatch_script(
    cmd: Command,
    stdout_path: str,
    stderr_path: str,
    rc_path: str,
    partition: str | None,
    wall_time: str,
) -> str:
    # `-l` per https://docs.cscs.ch/access/firecrest/ — CSCS clusters want a
    # login shell so module / environment defaults are sourced.
    sbatch_lines = [
        "#!/bin/bash -l",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --time={wall_time}",
    ]
    if partition:
        sbatch_lines.append(f"#SBATCH --partition={partition}")

    body: list[str] = []
    if cmd.working_directory is not None:
        body.append(f"cd {quote_path(cmd.working_directory)}")
    if cmd.env_vars:
        for k, v in cmd.env_vars.items():
            body.append(f"export {k}={shlex.quote(v)}")
    if isinstance(cmd.command, str):
        body.append(cmd.command)
    else:
        body.append(shlex.join(cmd.command))
    # Capture the command's own exit code — Slurm's job state tells us
    # COMPLETED vs FAILED but not the underlying number (127 vs 1 etc.).
    # The redirect on this line overrides Slurm's --output for this echo only.
    body.append(f"echo $? > {shlex.quote(rc_path)}")

    return "\n".join([*sbatch_lines, "", *body, ""])


def _extract_job_id(submit_resp: dict) -> str:
    # FirecREST v2 returns {"jobid": <int>} but older / wrapped responses have
    # been seen as {"jobId": ...} or {"job": {"jobid": ...}}. Try all three.
    for key in ("jobid", "jobId"):
        if key in submit_resp and submit_resp[key] is not None:
            return str(submit_resp[key])
    job = submit_resp.get("job")
    if isinstance(job, dict):
        for key in ("jobid", "jobId"):
            if key in job and job[key] is not None:
                return str(job[key])
    return ""


def _safe_view(client: f7t.v2.Firecrest, system_name: str, path: str, logger: Logger) -> str:
    # The CSCS gateway sometimes drops the keep-alive connection after the
    # long `wait_for_job` poll loop, so the very next `view` raises
    # `httpx.RemoteProtocolError` instead of returning data. One retry on a
    # fresh connection is enough in practice. A missing file is a
    # `FirecrestException` (404) — treat as empty.
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            return client.view(system_name=system_name, path=path)
        except f7t.FirecrestException as e:
            logger.warning(f"Could not read {path}: {e}")
            return ""
        except Exception as e:
            last_exc = e
            logger.warning(f"view({path}) attempt {attempt} failed transiently: {e}")
            # Force a fresh httpx session before retrying.
            try:
                client.close_session()
                client.create_new_session()
            except Exception:
                pass
    assert last_exc is not None
    raise last_exc
