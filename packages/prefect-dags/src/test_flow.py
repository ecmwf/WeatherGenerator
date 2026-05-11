from prefect import variables
import prefect
from weathergen.prefect_dags.cmd_runners import run_cmd, CmdContext, LocalContext, Command, CommandResult, GenericContext, EcmwfSshContext
from weathergen.prefect_dags.result import OpError, is_err, Result, unwrap
from weathergen.prefect_dags.slurm import SlurmJob, SlurmSubmissionResult, submit_slurm, get_slurm_job_states, await_completion
from weathergen.prefect_dags.prefect_wrapper import flow, task
from weathergen.prefect_dags.prefect_logging import get_run_logger
import logging
from weathergen.prefect_dags.prefect_logging import get_run_logger
from weathergen.prefect_dags.sbatch import sbatch_try, SlurmJobResult, sbatch

# ctx: CmdContext = LocalContext()
ctx: CmdContext = EcmwfSshContext(
    host="hpc-login",
)



@task(task_run_name="test_run_cmd")
async def test_run_cmd():
    logger = get_run_logger()
    cmd = Command(
        command="echo Hello, World! $HOME $TEST_VAR",
        env_vars={"TEST_VAR": "123"},
        working_directory=None,
    )
    result = await run_cmd(ctx, cmd, logger=logger)
    if is_err(result):
        logger.error(f"Command failed with error: {result.err}")
    else:
        logger.info(f"Command succeeded with stdout: {result.stdout} and stderr: {result.stderr}")


@task
async def get_pwd() -> str:
    logger = get_run_logger()
    cmd = Command(
        command=["pwd"],
    )
    result = unwrap(await run_cmd(ctx, cmd, logger=logger))
    return result.stdout.strip()

@task
async def test_slurm() -> Result[SlurmJobResult]:
    pwd:str = await get_pwd()
    res = await sbatch(
        ctx,
        job_name="test_job",
        command=["/home/ecm8774/work/WeatherGenerator/.venv/bin/python3",
                 "-c", "import time; time.sleep(20); print('hello')"],
        time_limit="00:01:00",
        working_directory=pwd,  
    )
    return res

@flow(log_prints=True, name="test_run_cmd")
async def test_run_cmd_flow(
    rerun_token: str | None = '4629513f-e0bd-42f6-8db2-d79965719217',
):
    t1 = test_slurm.submit()
    res = await test_slurm()
    print(res, type(res))
    res1 = t1.result()
    print(res1, type(res1))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_run_cmd_flow())