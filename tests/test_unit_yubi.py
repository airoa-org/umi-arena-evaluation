"""The YUBI transforms, against the layouts submission-format.html states.

Needs openpi importable (the image's openpi venv). Skipped elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openpi", reason="openpi not installed on the host")

from umi_arena.yubi.transforms import PUBLISH_DIM, YubiInputs, YubiOutputs, make_yubi_example  # noqa: E402


def _harness_obs() -> dict:
    """Inference-time observation: no action columns, native-resolution images."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    return {
        "observation.image.left": img,
        "observation.image.right": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation.pose.left_hand_root_to_right_hand_root.absolute":
            np.array([0.23, -0.05, 0.27, -0.58, -0.09, 0.72, 0.34], dtype=np.float32),
        "observation.joint_states": np.array([0.24, 0.78], dtype=np.float32),
        "prompt": "pre-warm",
    }


@pytest.mark.parametrize("with_center", [False, True])
def test_wrist_cameras_are_enabled_and_unused_base_slot_is_masked(with_center):
    obs = _harness_obs()
    if with_center:
        obs["observation.image.center"] = np.full((480, 640, 3), 127, dtype=np.uint8)
    out = YubiInputs()(obs)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert set(out["image_mask"]) == set(out["image"])
    assert out["image_mask"]["base_0_rgb"] == np.False_
    assert not out["image"]["base_0_rgb"].any()
    for wrist in ("left", "right"):
        assert out["image_mask"][f"{wrist}_wrist_0_rgb"] == np.True_
        np.testing.assert_array_equal(out["image"][f"{wrist}_wrist_0_rgb"], obs[f"observation.image.{wrist}"])
    assert "actions" not in out


def test_model_preprocessing_accepts_two_physical_cameras():
    from openpi.models import model

    out = YubiInputs()(_harness_obs())
    observation = model.Observation.from_dict({
        "image": {key: value[None] for key, value in out["image"].items()},
        "image_mask": {key: np.asarray([value]) for key, value in out["image_mask"].items()},
        "state": out["state"][None],
    })
    processed = model.preprocess_observation(None, observation, train=False)
    assert set(processed.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (1, 224, 224, 3) for image in processed.images.values())
    assert not processed.image_masks["base_0_rgb"].any()
    assert processed.image_masks["left_wrist_0_rgb"].all()
    assert processed.image_masks["right_wrist_0_rgb"].all()


def test_state_is_pose_then_joints_padded_to_action_dim():
    obs = _harness_obs()
    out = YubiInputs()(obs)
    assert out["state"].shape == (32,)
    np.testing.assert_array_equal(out["state"][:7], obs["observation.pose.left_hand_root_to_right_hand_root.absolute"])
    np.testing.assert_array_equal(out["state"][7:9], obs["observation.joint_states"])
    assert not out["state"][9:].any()


def test_training_actions_are_left_right_joints_padded():
    out = YubiInputs()(make_yubi_example())
    assert out["actions"].shape == (16, 32)
    assert not out["actions"][:, PUBLISH_DIM:].any()


def test_outputs_trim_to_the_publish_vector():
    out = YubiOutputs()({"actions": np.arange(32 * 32, dtype=np.float32).reshape(32, 32)})
    assert out["actions"].shape == (32, PUBLISH_DIM)
    assert out["actions"][0, 15] == 15
