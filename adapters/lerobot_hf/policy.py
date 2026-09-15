"""UMI Arena submission adapter — any LeRobot-family checkpoint.

The policy class is resolved from the checkpoint's own config.json, so this one
file covers smolvla, groot, pi0, pi05, act, diffusion — anything LeRobot
registers. It is strict about shape: the checkpoint must declare the two
cameras and the 9-dim state the contract sends, and emit 16 (or openpi's 32)
action columns. A checkpoint that does not is refused at load, with a message
that names the mismatch, rather than served with the wrong inputs.

Publish layout expected by the harness, 16 values per timestep:
    [ left.dx,  left.dy,  left.dz,  left.dqx,  left.dqy,  left.dqz,  left.dqw,
     right.dx, right.dy, right.dz, right.dqx, right.dqy, right.dqz, right.dqw,
     hand_left_motor_joint, hand_right_motor_joint ]
Both hands are PER-STEP DELTAS; the gripper values are ABSOLUTE joint positions.
Quaternions are xyzw. A fine-tune on the released UMI Arena dataset with
action = left(7) + right(7) + joints(2) emits exactly this.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors

PUBLISH_DIM = 16       # what the harness slices up
OPENPI_DIM = 32        # openpi-style padding; the publish vector is the first 16
STATE_DIM = 9          # inter-hand pose (7) + finger joints (2)
CAMERAS = ("observation.image.left", "observation.image.right")


def _check_features(cfg: dict) -> tuple[dict, int]:
    """Refuse a checkpoint whose declared features are not the contract's; cameras map by name suffix."""
    image_keys = [k for k in cfg["input_features"] if k.startswith("observation.images.")]
    state_dim = cfg["input_features"]["observation.state"]["shape"][0]
    action_dim = cfg["output_features"]["action"]["shape"][0]
    camera_for = {k: f"observation.image.{k.rsplit('.', 1)[-1]}" for k in image_keys}
    if sorted(camera_for.values()) != sorted(CAMERAS):
        raise ValueError(
            f"checkpoint declares {len(image_keys)} cameras {image_keys}; the contract sends "
            f"{list(CAMERAS)}, and the model's image keys must end in 'left' and 'right'")
    if state_dim != STATE_DIM:
        raise ValueError(
            f"checkpoint declares a {state_dim}-dim observation.state; the contract's state is "
            f"{STATE_DIM}: inter-hand pose (7) + finger joints (2)")
    if action_dim not in (PUBLISH_DIM, OPENPI_DIM):
        raise ValueError(
            f"checkpoint emits {action_dim} action columns; the contract needs "
            f"{PUBLISH_DIM} (or {OPENPI_DIM} with openpi padding)")
    return camera_for, action_dim


class Policy:
    def __init__(self, checkpoint_dir: str):
        self._dir = Path(checkpoint_dir)
        cfg = json.loads((self._dir / "config.json").read_text())

        self._camera_for, self._action_dim = _check_features(cfg)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls = get_policy_class(cfg["type"])
        self._model = cls.from_pretrained(str(self._dir)).to(self._device).eval()  # dtype: the checkpoint's own
        # Normalisation lives in the processors; the device override retargets a CPU-prepared pipeline.
        self._pre, self._post = make_pre_post_processors(
            self._model.config, str(self._dir),
            preprocessor_overrides=self._processor_overrides(),
        )

        print(f"[adapter] {cfg['type']}: cameras {list(self._camera_for)}, action {self._action_dim}", flush=True)
        self._prewarm()

    # --- observation mapping -------------------------------------------------

    def _image(self, arr: np.ndarray) -> torch.Tensor:
        """Native-resolution uint8 HWC RGB -> float CHW in [0,1], batched.
        No resize: every LeRobot policy resizes internally to its own size."""
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(self._device)
        return (t.permute(2, 0, 1).float() / 255.0).unsqueeze(0)

    def _batch(self, obs: dict) -> dict:
        batch = {k: self._image(obs[cam]) for k, cam in self._camera_for.items()}
        state = np.concatenate([
            np.asarray(obs["observation.pose.left_hand_root_to_right_hand_root.absolute"], np.float32),
            np.asarray(obs["observation.joint_states"], np.float32),
        ])
        batch["observation.state"] = torch.from_numpy(state).to(self._device).unsqueeze(0)
        batch["task"] = obs.get("prompt", "")
        return batch

    # --- action mapping ------------------------------------------------------

    def _to_publish_vector(self, actions: np.ndarray) -> np.ndarray:
        """Trim openpi-style 32-dim actions to the 16-column publish vector."""
        if actions.shape[-1] == OPENPI_DIM:
            return actions[:, :PUBLISH_DIM]
        if actions.shape[-1] != PUBLISH_DIM:
            raise ValueError(f"model emitted {actions.shape[-1]} action columns, expected {PUBLISH_DIM}")
        return actions

    def _prewarm(self) -> None:
        t0 = time.monotonic()
        self.infer(self._synthetic_observation())
        print(f"[adapter] pre-warmed in {time.monotonic() - t0:.1f}s", flush=True)

    @staticmethod
    def _synthetic_observation() -> dict:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        return {
            "observation.image.left": img,
            "observation.image.right": img.copy(),
            "observation.pose.left_hand_root_to_right_hand_root.absolute":
                np.array([0.23, -0.05, 0.27, -0.58, -0.09, 0.72, 0.34], dtype=np.float32),
            "observation.joint_states": np.array([0.24, 0.78], dtype=np.float32),
            "prompt": "pre-warm",
        }

    def infer(self, obs: dict) -> np.ndarray:
        batch = self._pre(self._batch(obs))
        with torch.no_grad():
            chunk = self._model.predict_action_chunk(batch)
        chunk = self._post(chunk)
        actions = chunk[0].detach().cpu().float().numpy()
        return self._to_publish_vector(actions).astype(np.float32)

    def _processor_overrides(self) -> dict:
        config = json.loads((self._dir / "policy_preprocessor.json").read_text())
        overrides = {}
        for step in config["steps"]:
            key = step.get("registry_name") or step.get("class", "").rsplit(".", 1)[-1]
            if key in ("device_processor", "DeviceProcessorStep"):
                overrides[key] = {"device": str(self._device)}
        return overrides
