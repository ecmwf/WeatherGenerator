#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "weathergen-prefect-dags",
# ]
#
# [tool.uv.sources]
# # Directly pull the package from Github.
# weathergen-prefect-dags = { git = "https://github.com/ecmwf/WeatherGenerator", branch = "tjh/dev/prefect-test", subdirectory = "packages/prefect-dags" }
# ///
from weathergen.prefect_dags import flow, run, sbatch
from weathergen.prefect_dags.cmd_runners import EcmwfSshContext

# The run context defines where the commands will be executed.
ctx = EcmwfSshContext(
    host="hpc-login",
)


@flow(log_prints=True)
def hello_world(rerun_token=None):
    command = "echo 'hello world'"
    # Run a command on the HPC:
    cmd_result = run(ctx, command=command)
    print(f"Command result: {cmd_result.stdout.strip()}")
    slurm_result = sbatch(
        ctx,
        job_name="hello_world_job",
        command=command,
        # You may need to adapt to your HPC, see test_flow.py on an example.
        working_directory="/tmp",
        time_limit="00:01:00",
        fetch_output=True,
    )
    print(f"Slurm job finished: {slurm_result.status}")
    print(f"Slurm job object: {slurm_result}")


if __name__ == "__main__":
    hello_world(rerun_token='flo12885ad9')
