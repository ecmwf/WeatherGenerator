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
from weathergen.prefect_dags import SlurmJobResult, flow, run, sbatch, task
from weathergen.prefect_dags.cmd_runners import EcmwfSshContext

# ctx: CmdContext = LocalContext()
# ctx: CmdContext = EcmwfSshContext(
#     host="santis",
#     account="ch17",
# )
# ctx: CmdContext = CscsFirecrestContext(
#     hpc="santis",
#     account="ch17",
#     consumer_key_path="~/.ssh/cscs_consumer_key",
#     consumer_secret_path="~/.ssh/cscs_consumer_secret",
# )
ctx = EcmwfSshContext(
    host="hpc-login",
)


@task
def get_pwd() -> str:
    return run(ctx, command=["pwd"]).stdout.strip()


@task(task_run_name="sleep_and_print-{sleep_sec}s")
def sleep_and_print(sleep_sec: int, pwd: str) -> SlurmJobResult:
    print(f"Working directory is {pwd}, sleeping for {sleep_sec} seconds...")
    res = sbatch(
        ctx,
        job_name=f"prefect_test_{sleep_sec}s",
        command=[
            "python3",
            "-c",
            f"import time; time.sleep({sleep_sec}); print('hello')",
        ],
        time_limit="00:01:00",
        working_directory=pwd,
    )
    print(f"result: {res}, type: {type(res)}")
    # assert sleep_sec < 6, "xxx"
    return res


@flow(log_prints=True)
def test_run_cmd_flow(
    rerun_token=None,
):
    # Get pwd on HPC
    pwd = get_pwd()
    print(f"Current working directory: {pwd}")
    sleep_times = [5, 10]
    # Submit all my jobs
    jobs = [sleep_and_print.submit(sleep_sec, pwd) for sleep_sec in sleep_times]
    # Wait for all the jobs to complete and print the results:
    for job in jobs:
        res = job.result()
        print(f"Job result: {res}, ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-token", default=None)
    args = parser.parse_args()
    test_run_cmd_flow(rerun_token=args.rerun_token)
