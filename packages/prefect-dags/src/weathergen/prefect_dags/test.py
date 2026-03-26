from dataclasses import dataclass
import logging
from prefect import flow, serve, task
from weathergen.common import logger
from weathergen.jsc_slurm_poller import slurm_queue_poller_jsc, slurm_queue_poller_cineca, HpcContext, run_command_on_hpc, CinecaContext
import random
from prefect.logging import get_run_logger
import os

@dataclass
class TrainingJobConfig:
    public_branch: str
    pulic_commit: str
    private_branch: str
    private_commit: str
    hpc: str

@task(retries=0, retry_delay_seconds=10, task_run_name="check_sanity")
async def check_sanity(config: TrainingJobConfig, ctx:HpcContext):
    await run_command_on_hpc(config.hpc, ctx, "ls $HOME", logger=get_run_logger())

@task(retries=0, retry_delay_seconds=10, task_run_name="checkout_code_public")
async def checkout_code_public(config: TrainingJobConfig, ctx:HpcContext):
    logger = get_run_logger()
    logger.info(f"Checking out code for public branch {config.public_branch} and commit {config.pulic_commit}")
    await run_command_on_hpc(config.hpc, ctx, f"cd $HOME/work/WeatherGenerator && git remote update && git checkout {config.public_branch} && git checkout {config.pulic_commit}", logger=logger   )
    # TODO: return the location of the repo


@task(retries=0, retry_delay_seconds=10, task_run_name="checkout_code_private")
async def checkout_code_private(config: TrainingJobConfig, ctx:HpcContext):
    logger = get_run_logger()
    logger.info(f"Checking out code for private branch {config.private_branch} and commit {config.private_commit}")
    await run_command_on_hpc(config.hpc, ctx, f"""
cd $HOME/work/WeatherGenerator-private
git remote update
git checkout {config.private_branch}
git checkout {config.private_commit}
 """, logger=logger   )

async def launch_slurm_job(config: TrainingJobConfig, ctx:HpcContext):
    logger = get_run_logger()
    logger.info(f"Launching slurm job on {config.hpc} with config: {config}")
    # TODO actually launch the job and return some info about it (e.g. job id)
    await run_command_on_hpc(config.hpc, ctx, """
cd $HOME/work/WeatherGenerator-private
./hpc/launch-slurm.py --time 10 --options "wgtags.org='ecmwf'" --options "wgtags.workflow='wg1224'" --options "wgtags.stage='test'" 
""", logger=logger)
 
@flow(log_prints=True, name="launch_training_job")
async def launch_training_job_flow():
    logger = get_run_logger()
    config = TrainingJobConfig(
        public_branch="develop",
        pulic_commit="0950f939e9425394d1b8a583911469f4af0bcddb",
        private_branch="main",
        private_commit="bb2f74a5f1d3b03f57fd109835d093bdbed94d7c",
        hpc="cineca",
    )
    logger.info(f"Launching training job with config: {config}")
    ctx: HpcContext = CinecaContext(username = "thunter0", ssh_key_path="/Users/tjhunter/.ssh/leonardo_key")
    await check_sanity(config, ctx)
    await checkout_code_public(config, ctx)
    await checkout_code_private(config, ctx)
    await launch_slurm_job(config, ctx)




if __name__ == "__main__":
    jsc_dep = slurm_queue_poller_jsc.to_deployment(
        name="slurm-queue-poller-jsc",
        tags=["infrastructure", "slurm-poller", "team:eng"],
    )
    cineca_dep = slurm_queue_poller_cineca.to_deployment(
        name="slurm-queue-poller-cineca",
        tags=["infrastructure", "slurm-poller", "team:eng"],
    )
    launch_training_job_flow_deployment = launch_training_job_flow.to_deployment(
        name="launch-training-job",
        tags=["training", "team:eng"],
    )
    serve(jsc_dep, cineca_dep, launch_training_job_flow_deployment)
