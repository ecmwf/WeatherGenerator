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

from weathergen.launch_slurm import launch_slurm, wait_for_completion
from weathergen.prefect_dags import SlurmJobResult, flow, run, sbatch_try, task
from weathergen.prefect_dags.cmd_runners import *  # noqa: F403
from weathergen.prefect_dags.result import is_err

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
ctx = SimpleSshContext(
    host="hpc-login",
)
# ctx = EcmwfEcaccessContext(
#     cert_path="~/.ecaccess_cert.crt",
# )
# ctx = CinecaSshContext(
#     hpc="leonardo",
#     user="thunter0"
# )


@task
def get_home() -> str:
    res = run(ctx, command="echo $HOME")
    assert res.stdout, "No output from pwd command"
    # ECMWF appends many other lines to the output, so we need to get the last one:
    last_line = res.stdout.strip().split("\n")[-1]
    print(f"last line of pwd output: '{last_line}'")
    return last_line


@task(task_run_name="sleep_and_print-{sleep_sec}s")
def sleep_and_print(sleep_sec: int, pwd: str) -> SlurmJobResult:
    print(f"Working directory is {pwd}, sleeping for {sleep_sec} seconds...")
    res1 = sbatch_try(
        ctx,
        job_name=f"prefect_test_{sleep_sec}s",
        command=[
            "python3",
            "-c",
            f"import time; time.sleep({sleep_sec}); print('hello')",
        ],
        time_limit="00:01:00",
        working_directory=pwd,
        fetch_output=True,
    )
    print(f"sbatch_try result: {res1}")
    assert not is_err(res1)
    if res1.status == "COMPLETED":
        print("Job succeeded as expected")
        return res1
    if res1.status == "TIMEOUT":
        print("Job timed out as expected")
        # TODO: continue with a 2nd job that runs the same command.
        return res1
    assert False, f"Job failed with unexpected status: {res1.status}"


@flow(log_prints=True)
def test_run_cmd_flow(
    rerun_token=None,
):
    # Get home directory on HPC
    home = get_home()
    print(f"Home directory: {home}")
    sleep_times = [1]
    # Submit all my jobs
    jobs = [sleep_and_print.submit(sleep_sec, home) for sleep_sec in sleep_times]
    # Wait for all the jobs to complete and print the results:
    for job in jobs:
        res = job.result()
        print(f"Job result: {res}, ")


@flow(log_prints=True)
def test_run_cmd_flow2(
    rerun_token=None,
):
    # wgp = "/users/thunter/work/WeatherGenerator-private/"
    working_dir = "~/work/"
    wgp = "./WeatherGenerator-private/"
    jobs = launch_slurm(
        ctx,
        wgp,
        working_dir=working_dir,
        stage="train",
        time="10",
        # base_config="WeatherGenerator/config/default_config.yml",
    )
    print(jobs)
    assert not is_err(jobs)
    final_status = wait_for_completion(ctx, jobs)
    print(final_status)


if __name__ == "__main__":
    test_run_cmd_flow2(rerun_token="my-experiment")
