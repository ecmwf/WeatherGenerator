"""
Utilities for running commands on a Slurm cluster using Prefect.
"""

from weathergen.prefect_dags.cmd_runners import Command, CommandResult, CommandRunner, CmdContext, get_command_runner, run_cmd
from weathergen.prefect_dags.result import OpError, Result, is_err
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SlurmJob:
    """
    Represents a Slurm job to be submitted.
    """
    job_name: str
    output: str | Path | None = None
    script_path: str | Path | None = None
    command: str | list[str] | None = None
    # Working directory of the *job* on the compute node (sbatch --chdir).
    working_directory: str | Path | None = None
    # Working directory of the sbatch *submission* process itself.
    submission_directory: str | Path | None = None
    slurm_options: dict[str, str] | None = None

@dataclass
class SlurmSubmissionResult:
    """
    Result of submitting a Slurm job, containing the job ID 
    and any relevant output.
    """

    # The job id
    # Represented as string to ensure JSON serialization does not overflow to floating point.
    job_id: str
    # The output path of the job.
    output: Path

async def submit_slurm(job: SlurmJob, ctx: CmdContext, logger: logging.Logger) -> Result[SlurmSubmissionResult]:
    """
    Submits a Slurm job using the provided context and logger.

    This function constructs the appropriate sbatch command based on the SlurmJob details and submits it using the run_cmd utility.

    Even if no output path is provided, this function will always construct a path.
    This is necessary to fully capture the final state of the job.

    Requirements:
    If the working directory and the submission directories are provided, they must be absolute.
    If the working directory is not provided, the script path must be absolute.
    If the output is not provided, the output will be: "{working_directory}/slum_job_{job_name}_%j.out" if working_directory is provided, or "{script_path_parent}/slurm_job_{job_name}_%j.out".


    It returns a SlurmSubmissionResult containing the job ID if submission is successful, or an OpError if there was an issue.
    """
    # Validate path requirements.
    if job.working_directory is not None and not Path(job.working_directory).is_absolute():
        return OpError(err=ValueError(f"working_directory must be absolute: {job.working_directory}"))
    if job.submission_directory is not None and not Path(job.submission_directory).is_absolute():
        return OpError(err=ValueError(f"submission_directory must be absolute: {job.submission_directory}"))
    if job.working_directory is None and job.script_path is not None and not Path(job.script_path).is_absolute():
        return OpError(err=ValueError(
            f"script_path must be absolute when working_directory is not set: {job.script_path}"
        ))

    # Resolve output path. Always populated so the final job state can be captured.
    if job.output is not None:
        output_path = Path(job.output)
    elif job.working_directory is not None:
        output_path = Path(job.working_directory) / f"slurm_job_{job.job_name}_%j.out"
    elif job.script_path is not None:
        output_path = Path(job.script_path).parent / f"slurm_job_{job.job_name}_%j.out"
    else:
        return OpError(err=ValueError(
            "Cannot determine output path: provide output, working_directory, or script_path"
        ))

    # Construct sbatch command.
    cmd_parts = ["sbatch", f"--output={output_path}"]

    if job.working_directory is not None:
        cmd_parts.append(f"--chdir={job.working_directory}")

    if job.slurm_options:
        for option, value in job.slurm_options.items():
            cmd_parts.append(f"--{option}={value}")

    if job.script_path:
        cmd_parts.append(str(job.script_path))
    elif job.command:
        cmd_parts.append("--wrap")
        cmd_parts.append(f"'{job.command}'")
    else:
        return OpError(err=ValueError("Either script_path or command must be provided in SlurmJob"))

    full_command = " ".join(cmd_parts)
    logger.info(f"Submitting Slurm job with command: {full_command}")

    cmd = Command(command=full_command, working_directory=job.submission_directory)
    result = await run_cmd(ctx, cmd, logger)
    logger.debug(f"Slurm submission command result: {result}")
    if is_err(result):
        return result

    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]
    # Resolve %j (and %J) in the output path now that we know the job id, so the
    # returned path points to the actual file Slurm will write.
    resolved_output = Path(str(output_path).replace("%j", job_id).replace("%J", job_id))
    return SlurmSubmissionResult(job_id=job_id, output=resolved_output)