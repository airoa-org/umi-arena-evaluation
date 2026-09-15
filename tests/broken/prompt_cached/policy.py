"""Broken on purpose: caches the first prompt and raises when it changes."""
import numpy as np


class Policy:
    def __init__(self, checkpoint_dir):
        self.prompt = None

    def infer(self, obs):
        if self.prompt is None:
            self.prompt = obs["prompt"]
        if obs["prompt"] != self.prompt:
            raise RuntimeError(f"prompt changed from {self.prompt!r} to {obs['prompt']!r}")
        return np.zeros((16, 16), dtype=np.float32)
