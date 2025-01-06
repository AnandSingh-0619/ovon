#!/bin/bash
#SBATCH --job-name=ovon-eval
#SBATCH --output=slurm_logs/eval/ovon-ddppo-%j.out
#SBATCH --error=slurm_logs/eval/ovon-ddppo-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --partition=kira-lab,overcap
#SBATCH --qos=short
#SBATCH --exclude=heistotron
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
export OVON_VIDEO_DIR=/nethome/asingh3064/flash/ovon/video_dir/ddppo_val_seen
split="val_unseen"

TENSORBOARD_DIR="tb/objectnav/eval/ddppo_${JOB_ID}"
CHECKPOINT_DIR="data/checkpoints/eval_od1/"
LOG_DIR="Logs/eval_ddppo_od_${JOB_ID}.log"

srun python -um ovon.run \
  --run-type eval \
  --exp-config config/experiments/rnn_rl.yaml \
  habitat_baselines.trainer_name="ddppo_od" \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_DIR} \
  habitat_baselines.log_file=${LOG_DIR} \
  habitat_baselines.rl.policy.name=PointNavResNetODPolicy \
  habitat_baselines.num_environments=1 \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/val_seen/val_seen.json.gz \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.video_option="['disk']" \
  habitat_baselines.video_dir=${OVON_VIDEO_DIR} \
  habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=720 \
  habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=1280 \
  habitat_baselines.rl.policy.obs_transforms.resize.size="[720, 720]" \
  habitat.seed=${RANDOM} \
  habitat.simulator.habitat_sim_v0.allow_sliding=False \

