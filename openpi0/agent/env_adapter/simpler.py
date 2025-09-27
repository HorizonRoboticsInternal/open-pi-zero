import json
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import torch

# from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transformers import AutoTokenizer

from openpi0.agent.env_adapter.base import BaseEnvAdapter
from openpi0.model.vla.processing import VLAProcessor
from openpi0.utils.geometry import euler2axangle, mat2euler, quat2mat

from hobot.robot_common.widowx250s_kinematics import WidowX250sKinematicsSolver
from scipy.spatial.transform import Rotation as R


class SimplerAdapter(BaseEnvAdapter):
    def __init__(
        self,
        dataset_statistics_path: str,
        pretrained_model_path: str,
        tokenizer_padding: str,
        num_image_tokens: int,
        image_size: Tuple[int, int],
        max_seq_len: int,
        action_normalization_type: str = "bound",
        proprio_normalization_type: str = "bound",
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.action_normalization_type = action_normalization_type
        self.proprio_normalization_type = proprio_normalization_type
        assert action_normalization_type in ["bound", "gaussian"]
        assert proprio_normalization_type in ["bound", "gaussian"]

        # for normalization
        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)

        # tokenizer and processer --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=tokenizer_padding,
        )

    def reset(self):
        pass


    def preprocess(
            self,
            obs: dict,
    ) -> dict:
        """Vectorized batch preprocess. Same instruction for all obs."""

        # Note that we need to use cv2 instead of torch.nn.functional.interpolate
        # This is because the Lanczos interpolation is not available in torch
        # and using bicubic instead results in a significant performance drop.
        old_imgs = obs["rgb"]
        imgs = []
        for o_img  in old_imgs:
            imgs.append(cv2.resize(
                    o_img,
                    self.image_size,
                    interpolation=cv2.INTER_LANCZOS4,
                ))
        imgs = torch.as_tensor(imgs, dtype=torch.uint8).permute(0, 3, 1, 2)

        # processor will handle batching if we give repeated instructions
        model_inputs = self.processor(
            text=obs["instrs"],
            images=imgs,
        )

        # stack proprios
        raw_proprios = self.preprocess_proprio(obs)

        if self.proprio_normalization_type == "bound":
            proprio = self.normalize_bound(
                raw_proprios,
                np.array(self.dataset_statistics["proprio"]["p01"]),
                np.array(self.dataset_statistics["proprio"]["p99"]),
                clip_min=-1,
                clip_max=1,
            )
        else:  # gaussian
            proprio = self.normalize_gaussian(
                raw_proprios,
                np.array(self.dataset_statistics["proprio"]["mean"]),
                np.array(self.dataset_statistics["proprio"]["std"]),
            )

        proprios = torch.as_tensor(proprio, dtype=torch.float32)[:, None, :]  # [B, T=1, dim]

        return {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "attention_mask": model_inputs["attention_mask"],
            "proprios": proprios,
        }


    def postprocess(
        self,
        actions: np.array,
    ) -> List[dict]:
        # gripper action is not normalized in training dataset
        if self.action_normalization_type == "bound":
            raw_actions_except_gripper = self.denormalize_bound(
                actions[..., :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                actions[..., :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )
        actions = np.concatenate(
            [
                raw_actions_except_gripper,
                self.postprocess_gripper(actions[..., -1:]),
            ],
            axis=2,
        )
        return actions

    def preprocess_proprio(self, obs: dict) -> np.array:
        raise NotImplementedError

    def postprocess_gripper(self, action: float) -> float:
        raise NotImplementedError

    def get_video_frame(self, env, obs: dict) -> np.array:
        """for recording video"""
        return get_image_from_maniskill2_obs_dict(env, obs)


class BridgeSimplerAdapter(SimplerAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

        self._kin_solver = WidowX250sKinematicsSolver()

    def reset(self):
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        # return raw_proprio
        jp = obs["joint_pos"]
        ee_tf = self._kin_solver.batched_fwd_kin(jp[:, :6])
        rm_bridge = ee_tf[:, :3, :3]
        rpy_bridge_converted = R.from_matrix(
            rm_bridge @ self.default_rot.T).as_euler("xyz")

        # TODO: need to properly respect gripper_closing_pos in
        # ManiSkill widow robot. Right now, we assume it is zero.
        gripper_opening = jp[:, -1:] / 0.04

        raw_proprio = np.concatenate([
            ee_tf[:, :3, 3],
            rpy_bridge_converted,
            gripper_opening,
        ], axis=1)
        return raw_proprio

    def postprocess_gripper(self, action: np.ndarray) -> np.ndarray:
        action_gripper = action > 0.5
        return action_gripper

class EDRSimplerAdapter(SimplerAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Constants
        self.sticky_gripper_num_repeat = 15  # same used in Octo. Note this is for every individual action, not every action chunk. Control freq is 3Hz, so roughly sticky for 5 seconds.

    def reset(self):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.previous_gripper_action = None
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal"""
        quat_xyzw = np.roll(obs["agent"]["eef_pos"][3:7], -1)
        gripper_width = obs["agent"]["eef_pos"][
            7
        ]  # from simpler, 0 for close, 1 for open
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                obs["agent"]["eef_pos"][:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 open, 1 close for simpler

        action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -action
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -action
        # self.previous_gripper_action = action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # sticky closing
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # reaching maximum sticky
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action
