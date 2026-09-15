"""openpi data config for the YUBI dataset.

Put this in your own ``TrainConfig`` to fine-tune with upstream openpi, and
build the same ``TrainConfig`` in ``policy.py`` to serve the result. Nothing is
registered in openpi's config registry.
"""

from collections.abc import Sequence
import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from umi_arena.yubi import transforms as yubi


@dataclasses.dataclass(frozen=True)
class LeRobotYubiDataConfig(DataConfigFactory):
    default_prompt: str | None = None

    # Concatenated per timestep by YubiInputs; the dataset's action.pose.*.relative columns hold the same numbers.
    action_sequence_keys: Sequence[str] = (
        "observation.pose.left_hand_root.relative",
        "observation.pose.right_hand_root.relative",
        "action.joint_states",
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.image.left": "observation.image.left",
                        "observation.image.right": "observation.image.right",
                        "observation.pose.left_hand_root_to_right_hand_root.absolute": "observation.pose.left_hand_root_to_right_hand_root.absolute",
                        "observation.pose.left_hand_root.relative": "observation.pose.left_hand_root.relative",
                        "observation.pose.right_hand_root.relative": "observation.pose.right_hand_root.relative",
                        "observation.joint_states": "observation.joint_states",
                        "action.joint_states": "action.joint_states",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[yubi.YubiInputs()],
            outputs=[yubi.YubiOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
