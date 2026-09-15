"""Broken on purpose: nan. A regression target for check.py."""
import numpy as np


def _nan():
    a = np.zeros((16, 16), dtype=np.float32)
    a[3, 5] = np.nan
    return a


class Policy:
    def __init__(self, checkpoint_dir):
        pass

    def infer(self, obs):
        return _nan()
