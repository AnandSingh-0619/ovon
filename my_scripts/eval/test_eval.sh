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

this_dir=$PWD

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
export MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
JOB_ID=${SLURM_JOB_ID}

source /nethome/asingh3064/flash/envs/etc/profile.d/conda.sh
conda deactivate
conda activate ovon

split="val_seen"

TENSORBOARD_DIR="tb/objectnav/train/"
CHECKPOINT_DIR="data/checkpoints/"
LOG_DIR="Logs/evalDaggerRL_${JOB_ID}.log"

srun python -um ovon.run \
  --run-type eval \
  --exp-config config/all/experiments/transformer_rl_ft_sparse_pirl.yaml \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_DIR} \
  habitat_baselines.log_file=${LOG_DIR} \
  habitat.dataset.data_path=data/datasets/ovon/hm3d/v2/val_seen/val_seen.json.gz \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.clip_objectgoal_sensor=clip_objectgoal_sensor \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache=siglip_seems_like_there_is_a_blank_ahead.pkl \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="OVON-v1" \
  habitat.simulator.type="OVONSim-v0" \
  habitat.task.measurements.distance_to_goal.type=OVONDistanceToGoal \
  habitat.simulator.habitat_sim_v0.allow_sliding=False \
  habitat_baselines.rl.policy.rgb_only=False \
  habitat_baselines.rl.ddppo.pretrained_weights=/coc/testnvme/asingh3064/ovon/data/checkpoints/DAgRL.pth \
  habitat_baselines.rl.ddppo.pretrained=True \
  habitat_baselines.rl.ppo.use_linear_lr_decay=True \
  habitat_baselines.rl.ppo.ppo_epoch=1 \
  habitat_baselines.rl.policy.fusion_type=concat \
  habitat_baselines.num_environments=16 \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.split=$split

