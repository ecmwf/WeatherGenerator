from weathergen.prefect_dags import SlurmJobResult, flow, run, sbatch, task
from weathergen.prefect_dags.cmd_runners import (
    CmdContext,
    EcmwfSshContext,
    CscsFirecrestContext
)
from pathlib import Path

# ctx: CmdContext = LocalContext()
# ctx: CmdContext = EcmwfSshContext(
#     host="hpc-login",
# )
ctx: CmdContext = CscsFirecrestContext(
    hpc = "santis",
    api_token="eyJ4NXQjUzI1NiI6Ik16QTVNMkUxT0RneE5qUTRaVGhrTWpNME1tWTVNR1pqT0RreU5UQTNZVFZtT0dVNU9ETmxNR1ExTnpoak9EUXhZVFkxTWpZME9XTmhaRE5oT1dWbU5BPT0iLCJraWQiOiJnYXRld2F5X2NlcnRpZmljYXRlX2FsaWFzIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ==.eyJzdWIiOiJ0aHVudGVyQGNhcmJvbi5zdXBlciIsImFwcGxpY2F0aW9uIjp7ImlkIjoxMTMwLCJ1dWlkIjoiNzlmNjUxNGMtMWQ2YS00Y2ZhLTkzZmUtYWY2ZWQ5OTcyYjE1In0sImlzcyI6Imh0dHBzOlwvXC9kZXZlbG9wZXIuc3ZjLmNzY3MuY2g6NDQzXC9vYXV0aDJcL3Rva2VuIiwia2V5dHlwZSI6IlBST0RVQ1RJT04iLCJwZXJtaXR0ZWRSZWZlcmVyIjoiIiwidG9rZW5fdHlwZSI6ImFwaUtleSIsInBlcm1pdHRlZElQIjoiIiwiaWF0IjoxNzc4NTk5MjQzLCJqdGkiOiI0MmNjMjJiNy0xMjAyLTQ4MDQtOTYzNy04Y2Y3NmM2OTI0M2MifQ==.fpXMUriNVaA81uiVUigSz85y6NDbBZsdDtxYzmi5_bxo1jU65V8dxd_orB5qYRygt1FGDW88gpq875D_nKXwnzvYa4rep10VmmOSv2AuFJe4ut_-HSB6r-9jNlUH3LnX7LCIp-Q8Jo7ntKxPWy7oGEi8HNGVXDQubv9UUDW17EQOsGkjgjr_WI7doeIjiQD7hKDseMjtG3wggfr757Ksd8MSeI75AbLE2hu3GfFB6Fvykv6_x8HI9FTUFiRm1DSqEFZRifmo8SMZOCW-kgdPTRe2n0w616RtW04gWTCOkcYNL4cVj4Oa3I5wEyWq8NMxUEOtexGVDWYcvjRKMGjBrw==",
    # api_token_path= Path.home() / ".ssh" / "firecrest",
)


@task
def ssl_mlp(lr:float, num_layers:int):
    launch_slurm(
        ctx,
        job_name="ssl_mlp",
        base_config="confip_jepa.yaml",
        options = {
            "losses.CrossEntropyLoss": {
                "lr": lr,
                "num_layers": num_layers,
            }
        }
    )

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

    for lr in [0.01, 0.001]:
        for num_layers in [2, 4]:
            ssl_mlp.submit(lr=lr, num_layers=num_layers)


if __name__ == "__main__":
    test_run_cmd_flow(rerun_token=None)
