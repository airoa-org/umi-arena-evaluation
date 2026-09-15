"""The practice suite rejects raw gripper values outside its selection rule."""

from types import SimpleNamespace

import numpy as np
import pytest

from umi_arena.replay_grippers import cutoff, validate_raw


@pytest.mark.parametrize("signal", ["observation", "action"])
def test_raw_check_catches_values_missing_from_metadata(signal):
    data = {s + ".joint_states": np.full((5, 2), -.1) for s in ("action", "observation")}
    episode = SimpleNamespace(data=data, length=5, metadata={"episode_index": 12})
    validate_raw(episode, -.1)
    data[signal + ".joint_states"][4, 1] = -.11
    with pytest.raises(ValueError, match="raw .* gripper below"):
        validate_raw(episode, -.1)


@pytest.mark.parametrize("value", [True, None, "0", float("nan"), float("inf"), .1])
def test_invalid_filter_cutoff(value):
    with pytest.raises(ValueError): cutoff(value)
