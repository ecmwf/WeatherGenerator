#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=develbooster
#SBATCH --output=/p/project1/hclimrep/vasireddy1/Wgen_lint_fix/WeatherGenerator/HPC_Scripts/%x.%j.out
#SBATCH --error=/p/project1/hclimrep/vasireddy1/Wgen_lint_fix/WeatherGenerator/HPC_Scripts/%x.%j.err

# important paths and directories (adapt if required!)
VENV_NAME=.venv
# BASE_DIR=${SLURM_SUBMIT_DIR}/../
VENV_DIR=/p/project1/hclimrep/vasireddy1/Wgen_lint_fix/WeatherGenerator
CONFIG_DIR=/p/project1/hclimrep/vasireddy1/WeatherGenerator_fcst_pretraining/WeatherGenerator/config

# Load basic modules from software stack
ml --force purge
ml use $OTHERSTAGES
ml Stages/2025

ml GCC/13.3.0
ml GCCcore/.13.3.0

ml OpenMPI/5.0.5

ml git/2.45.1
ml Python/3.12.3

# Activate virtual environment
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ${VENV_DIR}/${VENV_NAME}/bin/activate ]]; then
      #echo "Activating virtual environment ${VENV_DIR}/${VENV_NAME}..."
      echo "Run virtual env via uv."
      source ${VENV_DIR}/${VENV_NAME}/bin/activate
   else
      echo ${VENV_DIR}
      echo "ERROR: Requested virtual environment ${VENV_NAME} not found..."
      exit 1
   fi
fi

# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=TRACE
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# parameters for the validation
# k96gmzhl - exp4 - job 11432191
# muso9wvz - exp5 - job 11433915

run_id=k96gmzhl
epoch=8
start_date=202201010000
end_date=202201310000

# run job
# srun --label --cpu-bind=v --accel-bind=v train --private_config ${CONFIG_DIR}/paths.yml --config ${CONFIG_DIR}/pvt_cfg_pretrain_fcst_expt4.yaml
srun --label --cpu-bind=v --accel-bind=v evaluate --from_run_id ${run_id} --epoch ${epoch} -start ${start_date} -end ${end_date} --private_config ${CONFIG_DIR}/paths.yml

echo "Finished job."
date
