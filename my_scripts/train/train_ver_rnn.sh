#!/bin/bash
#SBATCH --job-name=ovon-train
#SBATCH --output=slurm_logs/ovon-ver-%j.out
#SBATCH --error=slurm_logs/ovon-ver-%j.err
#SBATCH --gpus a40:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --partition=kira-lab,overcap
#SBATCH --qos=short
#SBATCH --signal=USR1@100


export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export PYTHONPATH=~/flash/ovon/:$PYTHONPATH

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
JOB_ID=${SLURM_JOB_ID}

source /nethome/asingh3064/flash/envs/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

TENSORBOARD_DIR="tb/objectnav/train/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav/ver/"
LOG_DIR="Logs/evalDaggerRL_${JOB_ID}.log"

srun python -um ovon.run \
  --run-type train \
  --exp-config config/experiments/rnn_rl.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat_baselines.log_file=${LOG_DIR} \
  habitat_baselines.num_environments=24 
  # +habitat/task/lab_sensors@habitat.task.lab_sensors.clip_objectgoal_sensor=clip_objectgoal_sensor \
  # ~habitat.task.lab_sensors.objectgoal_sensor 