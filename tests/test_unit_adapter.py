"""The lerobot adapter's feature check, run on the host without torch or lerobot."""

from __future__ import annotations

from pathlib import Path

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / "adapters" / "lerobot_hf"
def _check_features():
    """The adapter's feature check, without its torch/lerobot imports."""
    src = (ADAPTER / "policy.py").read_text()
    ns: dict = {"PUBLISH_DIM": 16, "OPENPI_DIM": 32, "STATE_DIM": 9,
                "CAMERAS": ("observation.image.left", "observation.image.right")}
    exec(src[src.index("def _check_features"):src.index("class Policy")], ns)  # noqa: S102
    return ns["_check_features"]


def _cfg(cameras=("left", "right"), state=9, action=16) -> dict:
    return {"input_features": {**{f"observation.images.{c}": {} for c in cameras},
                               "observation.state": {"shape": [state]}},
            "output_features": {"action": {"shape": [action]}}}


@pytest.mark.parametrize("cfg,expect", [
    (_cfg(cameras=("left", "right", "center")), "declares 3 cameras"),
    (_cfg(cameras=("cam1", "cam2")), "must end in 'left' and 'right'"),
    (_cfg(state=6), "6-dim observation.state"),
    (_cfg(action=6), "emits 6 action columns"),
])
def test_feature_mismatches_are_refused_by_name(cfg, expect):
    with pytest.raises(ValueError, match=expect):
        _check_features()(cfg)


def test_matching_features_map_cameras_by_suffix():
    camera_for, action_dim = _check_features()(_cfg(cameras=("right", "left"), action=32))
    assert camera_for == {"observation.images.right": "observation.image.right",
                          "observation.images.left": "observation.image.left"}
    assert action_dim == 32

