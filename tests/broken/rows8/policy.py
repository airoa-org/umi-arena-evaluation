"""Broken on purpose: rows8. A regression target for check.py."""
import numpy as np



class Policy:
    def __init__(self, checkpoint_dir):
        pass

    def infer(self, obs):
        return np.zeros((8, 16), dtype=np.float32)
