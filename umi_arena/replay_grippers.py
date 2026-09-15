"""Validate the practice suite’s raw gripper references."""

import math


def cutoff(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0:
        raise ValueError("gripper minimum must be finite and at or below zero")
    return value



def validate_raw(episode, minimum):
    for signal in ("observation", "action"):
        values = episode.data[signal + ".joint_states"]
        if values.shape != (episode.length, 2) or any(not math.isfinite(v) for v in values.flat):
            raise ValueError("invalid raw gripper values")
        if values.min() < minimum:
            raise ValueError(f"episode {episode.metadata['episode_index']}: raw {signal} gripper below {minimum:g} rad")
