#!/bin/bash
#SBATCH --job-name=ovon-train
#SBATCH --output=slurm_logs/train/ovon-ddppo-%j.out
#SBATCH --error=slurm_logs/train/ovon-ddppo-%j.err
#SBATCH --gpus a40:16
#SBATCH --nodes 2
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --partition=overcap,kira-lab
#SBATCH --qos=short
#SBATCH --signal=USR1@100
#SBATCH --exclude=omgwth,clippy,major,gundam,heistotron,tachikoma,uniblab,randotron,chitti,spot,starrysky
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

TENSORBOARD_DIR="tb/objectnav/train/od_ddppo_${JOB_ID}"
CHECKPOINT_DIR="data/new_checkpoints/objectnav/od_ddppo_1092084"
LOG_DIR="Logs/ddppo_od_rnn_${JOB_ID}.log"
split="train"

srun python -um ovon.run \
  --run-type train \
  --exp-config config/experiments/rnn_rl.yaml \
  habitat_baselines.trainer_name="ddppo_od" \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat_baselines.log_file=${LOG_DIR} \
  habitat_baselines.rl.policy.name=PointNavResNetODPolicy \
  habitat_baselines.num_environments=32 \
  habitat_baselines.total_num_steps=500000000 \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/$split/$split.json.gz \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.writer_type="wb"