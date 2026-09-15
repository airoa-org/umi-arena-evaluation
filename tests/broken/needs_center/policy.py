"""Broken on purpose: depends on a camera the real robot does not send."""

import numpy as np


class Policy:
    def __init__(self, checkpoint_dir):
        pass

    def infer(self, obs):
        obs["observation.image.center"]
        return np.zeros((16, 16), dtype=np.float32)
