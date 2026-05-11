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
    res = run(
        ctx,
        command=["pwd"],
    )
    return res.stdout.strip()


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
    assert sleep_sec < 6, "xxx"
    return res


@flow(log_prints=True, name="test_run_cmd")
def test_run_cmd_flow(
    rerun_token: str | None = None,
):
    # Fan out two jobs concurrently via Prefect's task runner; no asyncio needed.
    t1_fut = sleep_and_print.submit(5)
    res = sleep_and_print(10)
    print(res, type(res))
    res1 = t1_fut.result()
    print(res1, type(res1))


if __name__ == "__main__":
    test_run_cmd_flow(rerun_token="0ca68051-7b0c-4baf-8c9b-ebe2c0aff7af")
