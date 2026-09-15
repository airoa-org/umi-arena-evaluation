"""openpi data transforms for the YUBI contract.

`YubiInputs` turns the harness observation into the pi0/pi0.5 input dict and,
at training time, builds the action chunk from the dataset columns.
`YubiOutputs` trims the model's padded action back to the 16-column publish
vector. Layouts are the ones `submission-format.html` sections 4 and 5 state.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms

PUBLISH_DIM = 16  # left delta pose (7) + right delta pose (7) + gripper joints (2)


def make_yubi_example() -> dict:
    """A random input example for the yubi policy: the dataset's column layout, images at the
    camera's native 480x640 as the harness sends them (the transforms resize)."""
    return {
        "observation.image.left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.image.right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.pose.left_hand_root_to_right_hand_root.absolute": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "observation.pose.left_hand_root.relative": np.zeros((16, 7), dtype=np.float32),
        "observation.pose.right_hand_root.relative": np.zeros((16, 7), dtype=np.float32),
        "observation.joint_states": np.zeros((2,), dtype=np.float32),
        "action.joint_states": np.zeros((16, 2), dtype=np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YubiInputs(transforms.DataTransformFn):
    """Inputs for the yubi policy.

    State = left-to-right relative pose (7) + finger joints (2), padded to ``action_dim``.
    Actions = concat(left.relative (7), right.relative (7), joints (2)) per timestep,
    padded to ``action_dim`` along the last axis.
    """

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["observation.image.left"])
        right_image = _parse_image(data["observation.image.right"])

        rel_pose = np.asarray(data["observation.pose.left_hand_root_to_right_hand_root.absolute"], dtype=np.float32)

        joints = np.asarray(data["observation.joint_states"], dtype=np.float32)
        state = np.concatenate([rel_pose, joints], axis=-1)
        state = transforms.pad_to_dim(state, self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(left_image),  # Upstream requires this slot even when the camera is unused.
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Training only: the per-step pose deltas are true deltas (dx dy dz dqx dqy dqz dqw).
        if "observation.pose.left_hand_root.relative" in data:
            left_act = np.asarray(data["observation.pose.left_hand_root.relative"], dtype=np.float32)
            right_act = np.asarray(data["observation.pose.right_hand_root.relative"], dtype=np.float32)
            joint_act = np.asarray(data["action.joint_states"], dtype=np.float32)
            actions = np.concatenate([left_act, right_act, joint_act], axis=-1)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim, axis=-1)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YubiOutputs(transforms.DataTransformFn):
    """Outputs for the yubi policy: a flat ``actions`` array, one row per timestep,
    laid out as ``[left_relative(7), right_relative(7), finger(2)]``."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, :PUBLISH_DIM]}  # drop the model's internal padding
