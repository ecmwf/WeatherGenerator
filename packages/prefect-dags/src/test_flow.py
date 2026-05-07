from prefect import flow, get_run_logger, task, variables
from prefect.logging import get_run_logger
from weathergen.prefect_dags.cmd_runners import run_cmd, CmdContext, LocalContext, Command, CommandResult, GenericContext, EcmwfSshContext
from weathergen.prefect_dags.result import OpError, is_err
import logging

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

@flow(log_prints=True, name="test_run_cmd")
async def test_run_cmd_flow():
    await test_run_cmd()
    logger = get_run_logger()
    cmd = Command(
        command=["ls", "-la", "."],
    )
    result = await run_cmd(ctx, cmd, logger=logger)
    if is_err(result):
        logger.error(f"Command failed with error: {result.err}")
    else:
        logger.info(f"Command succeeded with stdout: {result.stdout} and stderr: {result.stderr}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_run_cmd_flow())