"""Turn lerobot/smolvla_base into a 16-column, YUBI-shaped checkpoint.

The weights are untouched: SmolVLA pads state and action to 32 inside, so the
column count lives only in the feature shapes and the normalization stats. The
result loads as a YUBI submission and passes the contract check, but it has
never seen YUBI data. It is a fixture for the torch path, not a policy.

The shapes assume the training data was repacked into the two keys lerobot
reads, `observation.state` (9: inter-hand pose + finger joints) and `action`
(16: left delta pose, right delta pose, finger joints); the raw dataset stores
them as separate columns, and a stock lerobot-train on it yields a different
checkpoint.

Run inside the image's torch venv, on a GPU (the processors record the device):

    docker run --rm --gpus all -e HF_HOME=/hf/home -v ~/umi-arena-hf:/hf \\
      -v ~/umi-arena-hf/smolvla_base:/m:ro -v $PWD/scripts:/scripts:ro \\
      --entrypoint bash umi-arena-torch:<tag> -lc \\
      '/opt/venv/torch/bin/python /scripts/make_smolvla_yubi16.py /m /hf/smolvla_yubi16 /hf/yubi_stats.json'

`yubi_stats.json` is the dataset's `meta/stats.json`. It is not committed here,
because the dataset is gated; fetch it with your own access.
"""

import json
import sys

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

STATE_KEYS = ["observation.pose.left_hand_root_to_right_hand_root.absolute", "observation.joint_states"]
# The columns the YUBI training config takes the action from (umi_arena.yubi.config).
# The dataset's action.pose.*_hand_root.relative columns hold the same numbers.
ACTION_KEYS = ["observation.pose.left_hand_root.relative", "observation.pose.right_hand_root.relative", "action.joint_states"]


def main(src: str, dst: str, stats_path: str) -> None:
    stats = json.load(open(stats_path))

    def cat(keys):
        return {s: torch.tensor(sum((list(stats[k][s]) for k in keys), []), dtype=torch.float32)
                for s in ("mean", "std", "min", "max")}

    dataset_stats = {"observation.state": cat(STATE_KEYS), "action": cat(ACTION_KEYS)}
    assert dataset_stats["observation.state"]["mean"].shape == (9,)
    assert dataset_stats["action"]["mean"].shape == (16,)

    cfg = PreTrainedConfig.from_pretrained(src)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
    cfg.device = "cuda"

    SmolVLAPolicy.from_pretrained(src, config=cfg).save_pretrained(dst)
    pre, post = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
    pre.save_pretrained(dst)
    post.save_pretrained(dst)

    out = json.load(open(f"{dst}/config.json"))
    print("saved:", {k: v["shape"] for k, v in out["input_features"].items()},
          {k: v["shape"] for k, v in out["output_features"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
