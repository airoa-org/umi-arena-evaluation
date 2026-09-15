"""UMI Arena submission adapter — openpi / pi0.5.

Everything openpi-specific happens in __init__: the TrainConfig is built
in-process and handed to create_trained_policy, so the `config.py` registry is
never consulted and nothing is copied into the openpi tree.
"""

import time

import numpy as np
import openpi.models.pi0_config as pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

from umi_arena.openpi_yubi import LeRobotYubiDataConfig, discover_asset_id


class Policy:
    def __init__(self, checkpoint_dir: str):
        train_config = TrainConfig(
            name="umi_arena_submission",   # arbitrary; never looked up in a registry
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,             # pi0.5 trains with 32-dim actions
                action_horizon=32,
            ),
            data=LeRobotYubiDataConfig(
                # norm stats sit at <checkpoint>/assets/<asset_id>/; the id is found, not declared
                assets=AssetsConfig(asset_id=discover_asset_id(checkpoint_dir)),
                base_config=DataConfig(prompt_from_task=True),
            ),
            # weight_loader only matters when initialising training; inference
            # reads params straight from checkpoint_dir, so the default is right.
        )

        t0 = time.monotonic()
        self._policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
        print(f"[adapter] checkpoint loaded in {time.monotonic() - t0:.1f}s", flush=True)
        self._prewarm()

    def _prewarm(self) -> None:
        """The first inference JITs for longer than the websockets 20 s
        ping_timeout allows. Burn it here, before any client connects."""
        t0 = time.monotonic()
        self._policy.infer(self._synthetic_observation())
        print(f"[adapter] pre-warmed in {time.monotonic() - t0:.1f}s", flush=True)

    @staticmethod
    def _synthetic_observation() -> dict:
        """Shaped like what the harness sends — native-resolution uint8 HWC
        RGB, inter-hand pose, finger joints, prompt. The output is discarded, so
        only the shapes matter."""
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
        """obs arrives verbatim from the harness. Images are at the camera's
        native resolution; ResizeImages inside the transform pipeline handles it.

        Returns (chunk, 16) — YubiOutputs has already trimmed the model's internal
        32-dim padding to the publish vector:
          [left.dx dy dz dqx dqy dqz dqw, right.dx … dqw, hand_left, hand_right]
        """
        out = self._policy.infer(obs)
        return np.asarray(out["actions"], dtype=np.float32)   # openpi returns float64
