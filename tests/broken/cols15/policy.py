"""Broken on purpose: cols15. A regression target for check.py."""
import numpy as np



class Policy:
    def __init__(self, checkpoint_dir):
        pass

    def infer(self, obs):
        return np.zeros((16, 15), dtype=np.float32)
