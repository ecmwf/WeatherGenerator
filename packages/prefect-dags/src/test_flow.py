from weathergen.prefect_dags import run, sbatch, task, flow, SlurmJobResult
from weathergen.prefect_dags.cmd_runners import (
    CmdContext,
    EcmwfSshContext,
)

# ctx: CmdContext = LocalContext()
ctx: CmdContext = EcmwfSshContext(
    host="hpc-login",
)


@task
def get_pwd() -> str:
    return run(
        ctx,
        command=["pwd"],
    ).stdout.strip()


@task
def sleep_and_print(sleep_sec: int) -> SlurmJobResult:
    pwd = get_pwd()
    print(f"Working directory is {pwd}, sleeping for {sleep_sec} seconds...")
    res = sbatch(
        ctx,
        job_name="test_job",
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


@flow(log_prints=True, name="test_run_cmd")
def test_run_cmd_flow(
    rerun_token: str | None = None,
):
    # Fan out two concurrent jobs.
    job1 = sleep_and_print.submit(5)
    job2 = sleep_and_print.submit(10)
    res1, res2 = job1.result(), job2.result()
    print(res1)
    print(res2)


if __name__ == "__main__":
    test_run_cmd_flow(rerun_token=None)
