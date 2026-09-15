"""Turn nvidia/GR00T-N1.7-3B into a 16-column, YUBI-shaped lerobot checkpoint.

The weights are untouched: GR00T pads state and action to 132 inside, and the
`new_embodiment` head is what a fine-tune on a new robot starts from. Only the
feature shapes and the normalization stats are set. The result loads as a YUBI
submission and passes the contract check, but it has never seen YUBI data. It
is a fixture for the GR00T path, not a policy.

Run inside the torch image, on a GPU:

    docker run --rm --gpus all -e HF_HOME=/hf/home -v ~/umi-arena-hf:/hf \\
      -v $PWD/scripts:/scripts:ro --entrypoint bash umi-arena-torch:<tag> -lc \\
      '/opt/venv/torch/bin/python /scripts/make_groot_yubi16.py nvidia/GR00T-N1.7-3B /hf/groot_yubi16 /hf/yubi_stats_groot.json'

`yubi_stats_groot.json` holds min/max/mean/std/q01/q99 for `observation.state`
(9) and `action` (16), derived from the dataset with the columns
`umi_arena.yubi.config` trains from: pose (7) + `observation.joint_states` (2), and
`observation.pose.{left,right}_hand_root.relative` (7+7) + `action.joint_states` (2). It is not committed here,
because the dataset is gated; derive it with your own access.
"""

import json
import sys

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy


def main(base: str, dst: str, stats_path: str) -> None:
    stats = json.load(open(stats_path))
    dataset_stats = {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()} for k, d in stats.items()}
    assert dataset_stats["observation.state"]["mean"].shape == (9,)
    assert dataset_stats["action"]["mean"].shape == (16,)

    cfg = GrootConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
            "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        base_model_path=base,
        embodiment_tag="new_embodiment",
        chunk_size=16,
        n_action_steps=16,
        device="cuda",
    )

    GrootPolicy.from_pretrained(base, config=cfg).save_pretrained(dst)
    pre, post = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
    pre.save_pretrained(dst)
    post.save_pretrained(dst)

    out = json.load(open(f"{dst}/config.json"))
    print("saved:", {k: v["shape"] for k, v in out["input_features"].items()},
          {k: v["shape"] for k, v in out["output_features"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
