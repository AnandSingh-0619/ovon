import numpy as np
import torch
from gym import spaces
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from ovon.utils.rollout_storage_no_2d import RolloutStorageNo2D
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_num_actions,
    inference_mode,
    is_continuous_action_space,
)
import time
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat import VectorEnv, logger
from omegaconf import OmegaConf
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from collections import defaultdict, deque
import os
import tqdm
from habitat.utils.visualizations.utils import observations_to_image
from habitat.utils.render_wrapper import overlay_frame
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay

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
                self.agent.actor_critic.net.object_detector
            )
        if self._use_ovod:
            batch[self.agent.actor_critic.net.SEG_MASKS] = self._yolo_detector.predict(batch)
            
        obs_space = spaces.Dict(
            {
                self.agent.actor_critic.net.SEG_MASKS: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=self._yolo_detector.output_shape,
                    dtype=np.float32,
                ),
                **obs_space.spaces,
            }
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
                batch[self.agent.actor_critic.net.SEG_MASKS] = self._yolo_detector.predict(batch)
            
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start
    
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )

        ppo_cfg = config.habitat_baselines.rl.ppo

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split
            config.habitat.dataset.split='val_seen'
            config.habitat.dataset.data_path='data/datasets/ovon/hm3d/val_seen/val_seen.json.gz'
            config.habitat_baselines.total_num_steps=500000000
            

        if (
            len(config.habitat_baselines.video_render_views) > 0
            and len(self.config.habitat_baselines.eval.video_option) > 0
        ):
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            render_view_uuids = [
                agent_sensors[render_view].uuid
                for render_view in config.habitat_baselines.video_render_views
                if render_view in agent_sensors
            ]
            assert len(render_view_uuids) > 0, (
                f"Missing render sensors in agent config: "
                f"{config.habitat_baselines.video_render_views}."
            )
            with read_write(config):
                for render_view_uuid in render_view_uuids:
                    if render_view_uuid not in config.habitat.gym.obs_keys:
                        config.habitat.gym.obs_keys.append(render_view_uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):
            # Assume NONE of the actions are discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.habitat_baselines.num_environments,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = (
            self.config.habitat_baselines.test_episode_count
        )
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions.cpu()
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            policy_info = self.actor_critic.get_policy_info(infos, dones)
            for i in range(len(policy_info)):
                infos[i].update(policy_info[i])
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                    frame = overlay_frame(frame, infos[i])
                    rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if (
                        len(self.config.habitat_baselines.eval.video_option)
                        > 0
                    ):
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
