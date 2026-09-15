"""Broken on purpose: each call sleeps longer than the keepalive (UMI_CHECK_SLEEP_S, default 5)."""
import os
import time

import numpy as np


class Policy:
    def __init__(self, checkpoint_dir):
        self.sleep_s = float(os.environ.get("UMI_CHECK_SLEEP_S", "5"))

    def infer(self, obs):
        time.sleep(self.sleep_s)
        return np.zeros((16, 16), dtype=np.float32)
