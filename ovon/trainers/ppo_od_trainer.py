import numpy as np
import torch
from gym import spaces
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet

from ovon.utils.rollout_storage_no_2d import RolloutStorageNo2D
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_num_actions,
    inference_mode,
    is_continuous_action_space,
)
import time

@baseline_registry.register_trainer(name="ddppo_od")
@baseline_registry.register_trainer(name="ppo_od")
class PPO_ODTrainer(PPOTrainer):
    def _init_train(self, *args, **kwargs):
        super()._init_train(*args, **kwargs)
        # Hacky overwriting of existing RolloutStorage with a new one
        ppo_cfg = self.config.habitat_baselines.rl.ppo
        action_shape = self.rollouts.buffers["actions"].shape[2:]
        discrete_actions = self.rollouts.buffers["actions"].dtype == torch.long
        batch = self.rollouts.buffers["observations"][0]

        obs_space = spaces.Dict({
            PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=self._encoder.output_shape,
                dtype=np.float32,
            ),
            **self.obs_space.spaces,
        })
        # self._use_ovod = self.config.habitat_baselines.rl.ddppo.use_detector
        self._use_ovod=True
        if self._use_ovod:
            self._yolo_detector = (
                self._agent.actor_critic.net.object_detector
            )
        if self._use_ovod:
            batch[self._agent.actor_critic.net.SEG_MASKS] = self._yolo_detector.predict(
                batch
            )
        self.rollouts = RolloutStorageNo2D(
            self.actor_critic.net.visual_encoder,
            batch,
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)


    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )
            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with inference_mode():
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)
        if self._use_ovod:
            with torch.no_grad():
                self._masks = self._yolo_detector.predict(batch)

            batch[self._agent.actor_critic.net.SEG_MASKS] = self._masks
            
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start