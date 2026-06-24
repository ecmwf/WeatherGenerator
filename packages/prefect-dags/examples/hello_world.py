#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "weathergen-prefect-dags",
# ]
#
# [tool.uv.sources]
# # Directly pull the package from Github.
# # weathergen-prefect-dags = { git = "https://github.com/ecmwf/WeatherGenerator", branch = "tjh/dev/prefect-test", subdirectory = "packages/prefect-dags" }
#
# # When developing locally, swap the source above for the line below:
# weathergen-prefect-dags = { path = "../", editable = true }
# ///
from weathergen.prefect_dags import flow, run, sbatch
from weathergen.prefect_dags.cmd_runners import SimpleSshContext

# The run context defines where the commands will be executed.
ctx = SimpleSshContext(
    host="hpc-login",
)


@flow(log_prints=True)
def hello_world(rerun_token=None):
    # The command to run on the HPC:
    command = "echo 'hello world'"

    # Run a command interactively:
    cmd_result = run(ctx, command=command)
    print(f"Command result: {cmd_result.stdout.strip()}")

    # Run a command on the HPC using sbatch:
    slurm_result = sbatch(
        ctx,
        job_name="hello_world_job",
        command=command,
        working_directory="~",
        time_limit="00:01:00",
        fetch_output=True,
    )
    print(f"Slurm job finished: {slurm_result.status}")
    print(f"Slurm job object: {slurm_result}")


if __name__ == "__main__":
    hello_world(rerun_token=None)
