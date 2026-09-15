"""Dummy submission adapter — no model, just a well-formed action chunk.

Proves the contract end to end without touching the GPU. Emits a small constant
per-step delta so the recorded action topics carry non-zero, identifiable values.
"""

import numpy as np

CHUNK = 16
ACTION_DIM = 16


class Policy:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        # [dx dy dz qx qy qz qw] per hand + 2 gripper joints.
        step = np.zeros(ACTION_DIM, dtype=np.float32)
        step[0] = 1e-3          # left.dx
        step[6] = 1.0           # left.dqw  (identity rotation delta)
        step[7] = -1e-3         # right.dx
        step[13] = 1.0          # right.dqw
        step[14] = 0.7          # hand_left_motor_joint  (absolute)
        step[15] = 0.7          # hand_right_motor_joint (absolute)
        self._chunk = np.tile(step, (CHUNK, 1))

    def infer(self, obs: dict) -> np.ndarray:
        return self._chunk.copy()
