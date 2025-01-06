from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn

from ovon.models.encoders.cma_xattn import CrossModalAttention
from ovon.models.encoders.cross_attention import CrossAttention
from ovon.models.encoders.make_encoder import make_encoder
from ovon.task.sensors import ClipObjectGoalSensor
from ovon.models.detection.yolo_ow import YOLOPerception
from ovon.models.encoders.mask_encoder import mask_encoder
if TYPE_CHECKING:
    from omegaconf import DictConfig
from ovon.models.clip_policy import PointNavResNetCLIPPolicy, OVONNet, FusionType


@baseline_registry.register_policy
class PointNavResNetODPolicy(PointNavResNetCLIPPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        backbone: str = "clip_avgpool",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        depth_ckpt: str = "",
        fusion_type: str = "cross_attention",
        attn_heads: int = 3,
        use_vis_query: bool = True,
        use_residual: bool = True,
        residual_vision: bool = True,
        unfreeze_xattn: bool = False,
        rgb_only: bool = True,
        use_prev_action: bool = True,
        use_odom: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
            depth_ckpt=depth_ckpt,
            fusion_type=fusion_type,
            attn_heads=attn_heads,
            use_vis_query=use_vis_query,
            use_residual=use_residual,
            residual_vision=residual_vision,
            unfreeze_xattn=unfreeze_xattn,
            rgb_only=rgb_only,
            use_prev_action=use_prev_action,
            use_odom=use_odom, 
            **kwargs)
        
        self.unfreeze_xattn = unfreeze_xattn
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
            
        self.net = OVONNetOD(
            observation_space=observation_space,
            action_space=action_space,  # for previous action
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            discrete_actions=discrete_actions,
            depth_ckpt=depth_ckpt,
            fusion_type=fusion_type,
            attn_heads=attn_heads,
            use_vis_query=use_vis_query,
            use_residual=use_residual,
            residual_vision=residual_vision,
            rgb_only=rgb_only,
            use_prev_action=use_prev_action,
            use_odom=use_odom,
            **kwargs)
        
        
class OVONNetOD(Net):
    SEG_MASKS = "segmentation_masks"

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone: str,
        discrete_actions: bool = True,
        fusion_type: str = "concat",
        clip_embedding_size: int = 1024,  # Target category CLIP embedding size
        attn_heads: int = 3,
        use_vis_query: bool = False,
        use_residual: bool = True,
        residual_vision: bool = False,
        rgb_only: bool = True,
        use_prev_action: bool = True,
        use_odom: bool = False,
        *args,
        **kwargs,
    ):
        print("Observation space info:")
        for k, v in observation_space.spaces.items():
            print(f"  {k}: {v}")

        super().__init__()
        self.discrete_actions = discrete_actions
        self._fusion_type = FusionType(fusion_type)
        self._hidden_size = hidden_size
        self._rgb_only = rgb_only

        self._use_prev_action = not (rgb_only or not use_prev_action)
        self._use_odom = not (rgb_only or not use_odom)

        # Embedding layer for previous action
        self._n_prev_action = 32
        rnn_input_size_info = {}
        if self._use_prev_action:
            rnn_input_size_info["prev_action"] = self._n_prev_action
            if discrete_actions:
                self.prev_action_embedding = nn.Embedding(
                    action_space.n + 1, self._n_prev_action
                )
            else:
                num_actions = get_num_actions(action_space)
                self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)

        # Object Detection
        self.object_detector = YOLOPerception()
        rnn_input_size_info["mask_embedding"] = 768
        
        # Visual encoder
        self.visual_encoder = make_encoder(backbone, observation_space)
        if backbone in ["clip_attnpool", "siglip"]:
            self.visual_fc = nn.Identity()
            if backbone == "clip_attnpool":
                clip_embedding_size = 1024
            else:
                clip_embedding_size = 768
            visual_feats_size = clip_embedding_size
        else:
            if backbone == "resnet":
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                    nn.ReLU(True),
                )
            else:
                self.visual_fc = nn.Sequential(
                    nn.Linear(self.visual_encoder.output_size, hidden_size),
                    nn.ReLU(True),
                )
            visual_feats_size = hidden_size

        # Optional Compass embedding layer
        if (
            EpisodicCompassSensor.cls_uuid in observation_space.spaces
            and self._use_odom
        ):
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size_info["compass_embedding"] = 32

        # Optional GPS embedding layer
        if EpisodicGPSSensor.cls_uuid in observation_space.spaces and self._use_odom:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size_info["gps_embedding"] = 32

        # Optional cross-attention layer
        if self._fusion_type.concat:
            rnn_input_size_info["visual_feats"] = visual_feats_size
        elif self._fusion_type.xattn:
            if backbone in ["clip_attnpool", "siglip"]:
                if backbone == "clip_attnpool":
                    embed_dim = 1024
                else:
                    embed_dim = 768
                assert clip_embedding_size == embed_dim
                assert visual_feats_size == embed_dim
            else:
                embed_dim = None
            if self._fusion_type.cma:
                self.cross_attention = CrossModalAttention(
                    text_embedding_dim=clip_embedding_size,
                    rgb_embedding_dim=visual_feats_size,
                    hidden_size=512,
                )
            else:
                self.cross_attention = CrossAttention(
                    x1_dim=clip_embedding_size,
                    x2_dim=visual_feats_size,
                    num_heads=attn_heads,
                    use_vis_query=use_vis_query,
                    use_residual=use_residual,
                    residual_vision=residual_vision,
                    embed_dim=embed_dim,
                )
            rnn_input_size_info["visual_feats"] = self.cross_attention.output_size
        else:
            raise NotImplementedError(f"Unknown fusion type: {fusion_type}")

        assert ClipObjectGoalSensor.cls_uuid in observation_space.spaces
        if not self._fusion_type.late_fusion and not self._fusion_type.xattn:
            rnn_input_size_info["clip_goal"] = clip_embedding_size

        # Report the type and sizes of the inputs to the RNN
        self.rnn_input_size = sum(rnn_input_size_info.values())
        print("RNN input size info: ")
        for k, v in rnn_input_size_info.items():
            print(f"  {k}: {v}")
        total_str = f"  Total RNN input size: {self.rnn_input_size}"
        print("  " + "-" * (len(total_str) - 2) + "\n" + total_str)

        self.rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        self.state_encoder = self.build_state_encoder()

        print(
            "State encoder parameters: ",
            sum(p.numel() for p in self.state_encoder.parameters()),
        )

        if self._fusion_type.late_fusion:
            self.late_fusion_fc = nn.Sequential(
                nn.Linear(clip_embedding_size, hidden_size),
                nn.ReLU(True),
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def build_state_encoder(self):
        return build_rnn_state_encoder(
            self.rnn_input_size,
            self._hidden_size,
            rnn_type=self.rnn_type,
            num_layers=self._num_recurrent_layers,
        )     
       

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        # We CANNOT use observations.get() here because
        # self.visual_encoder(observations) is an expensive operation. Therefore,
        # we need `# noqa: SIM401`
        if (  # noqa: SIM401
            PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observations
        ):
            visual_feats = observations[
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
            ]
        else:
            visual_feats = self.visual_encoder(observations)

        visual_feats = self.visual_fc(visual_feats)
        object_goal = observations[ClipObjectGoalSensor.cls_uuid]

        if self._fusion_type.xattn:
            visual_feats = self.cross_attention(object_goal, visual_feats)

        x = [visual_feats]

        if self._fusion_type.concat and not self._fusion_type.late_fusion:
            x.append(object_goal)
            
        if self.SEG_MASKS in observations:
            mask_feats = observations[self.SEG_MASKS]
        else:
            mask_feats = self.object_detector.predict(observations)
            
        x.append(mask_feats)        
        if EpisodicCompassSensor.cls_uuid in observations and self._use_odom:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations and self._use_odom:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        if self._use_prev_action:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks[:, -1:].view(-1), prev_actions + 1, start_token)
            )

            x.append(prev_actions)

        out = torch.cat(x, dim=1)

        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )

        if self._fusion_type.late_fusion:
            out = (out + visual_feats) * self.late_fusion_fc(object_goal)

        aux_loss_state = {
            "rnn_output": out,
            "perception_embed": visual_feats,
        }

        return out, rnn_hidden_states, aux_loss_state
