#!/bin/bash
#SBATCH --job-name=ovon-eval
#SBATCH --output=slurm_logs/eval/ovon-ver-%j.out
#SBATCH --error=slurm_logs/eval/ovon-ver-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --partition=kira-lab,overcap
#SBATCH --qos=short
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
JOB_ID=${SLURM_JOB_ID}

source /nethome/asingh3064/flash/envs/etc/profile.d/conda.sh
conda deactivate
conda activate ovonv3

TENSORBOARD_DIR="tb/objectnav/eval/ddppo_${JOB_ID}"
CHECKPOINT_DIR="data/checkpoints/eval/"
LOG_DIR="Logs/ddppo_rnn_${JOB_ID}.log"
split="val_unseen"

srun python -um ovon.run \
  --run-type eval \
  --exp-config config/experiments/rnn_rl.yaml \
  habitat_baselines.trainer_name="ddppo_no_2d" \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_DIR} \
  habitat_baselines.log_file=${LOG_DIR} \
  habitat_baselines.num_environments=24 \
   habitat_baselines.load_resume_state_config=True 
 