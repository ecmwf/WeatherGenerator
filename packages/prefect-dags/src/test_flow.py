
from weathergen.prefect_dags import SlurmJobResult, flow, run, sbatch, task
from weathergen.prefect_dags.cmd_runners import CmdContext, CscsFirecrestContext

# ctx: CmdContext = LocalContext()
# ctx: CmdContext = EcmwfSshContext(
#     host="hpc-login",
# )
ctx: CmdContext = CscsFirecrestContext(
    hpc="santis",
    account="ch17",
    consumer_key="eb1f48aa-3317-44e9-8316-38dde71c3f94",
    consumer_secret="vaSIryqsgq6VJ2oR7AMXSPhykFpAEg9e",
)


# @task
# def ssl_mlp(lr: float, num_layers: int):
#     launch_slurm(
#         ctx,
#         job_name="ssl_mlp",
#         base_config="confip_jepa.yaml",
#         options={
#             "losses.CrossEntropyLoss": {
#                 "lr": lr,
#                 "num_layers": num_layers,
#             }
#         },
#     )


@task
def get_pwd() -> str:
    return run(ctx, command=["pwd"]).stdout.strip()


@task
def sleep_and_print(sleep_sec: int, pwd: str) -> SlurmJobResult:
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


@flow(log_prints=True)
def test_run_cmd_flow(
    rerun_token: str | None = None,
):
    # Get pwd on HPC
    pwd = get_pwd()
    print(f"Current working directory: {pwd}")
    # Fan out two concurrent jobs.
    job1 = sleep_and_print.submit(5, pwd)
    job2 = sleep_and_print.submit(10, pwd)
    res1, res2 = job1.result(), job2.result()
    print(res1)
    print(res2)

    # for lr in [0.01, 0.001]:
    #     for num_layers in [2, 4]:
    #         ssl_mlp.submit(lr=lr, num_layers=num_layers)


if __name__ == "__main__":
    test_run_cmd_flow(rerun_token=None)
