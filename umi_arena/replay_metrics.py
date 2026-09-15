"""Pose and action comparisons in physical units."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


def poses(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 7 or not np.isfinite(values).all():
        raise ValueError("poses must contain seven finite values")
    norms = np.linalg.norm(values[..., 3:], axis=-1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-8):
        raise ValueError("zero or unbounded quaternion cannot describe a pose")
    return values


def delta_between(before, after, translation_frame):
    before, after = poses(before), poses(after)
    rotation = Rotation.from_quat(before[..., 3:])
    translation = after[..., :3] - before[..., :3]
    if translation_frame == "body":
        translation = rotation.inv().apply(translation)
    elif translation_frame != "world":
        raise ValueError("translation frame must be body or world")
    delta = rotation.inv() * Rotation.from_quat(after[..., 3:])
    return np.concatenate([translation, delta.as_quat(canonical=True)], axis=-1)


def integrate(anchor, deltas, translation_frame):
    current = poses(anchor).copy()
    result = []
    for delta in poses(deltas):
        rotation = Rotation.from_quat(current[3:])
        translation = delta[:3]
        if translation_frame == "body":
            translation = rotation.apply(translation)
        elif translation_frame != "world":
            raise ValueError("translation frame must be body or world")
        current = np.r_[current[:3] + translation,
                        (rotation * Rotation.from_quat(delta[3:])).as_quat(canonical=True)]
        result.append(current)
    return np.asarray(result)


def pose_error(predicted, reference):
    predicted, reference = poses(predicted), poses(reference)
    position = np.linalg.norm(predicted[..., :3] - reference[..., :3], axis=-1) * 100
    relative = Rotation.from_quat(predicted[..., 3:]).inv() * Rotation.from_quat(reference[..., 3:])
    return position, np.rad2deg(relative.magnitude())


@dataclass(frozen=True)
class Settings:
    action_hz: float
    translation_frame: str
    pose_timing: str
    position_cm: float = 2.0  # Tolerances drive the secondary within-tolerance percentage.
    rotation_deg: float = 5.0
    gripper_rad: float = 0.05
    chunk_size: int = field(default=16, init=False)  # The evaluation consumes 16 actions per call.
    position_scale_cm: float = 2.0  # Scales normalize the errors for headline RMSE.
    rotation_scale_deg: float = 5.0
    gripper_scale_rad: float = 0.05

    def __post_init__(self):
        for name in ("action_hz", "position_cm", "rotation_deg", "gripper_rad",
                     "position_scale_cm", "rotation_scale_deg", "gripper_scale_rad"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.translation_frame not in ("world", "body") or self.pose_timing not in ("previous", "next"):
            raise ValueError("specify translation_frame (world/body) and pose_timing (previous/next)")

    def stride(self, fps):
        ratio = fps / self.action_hz
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("dataset fps must be an integer multiple of action_hz")
        return int(round(ratio))


def compare(actions, anchors, references, grippers, settings):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16 or len(actions) < settings.chunk_size:
        raise ValueError("policy must return an (N, 16) action chunk with N >= 16")
    if not np.isfinite(actions).all():
        raise ValueError("policy returned non-finite actions")
    count = len(grippers)
    # Unused finite rows may contain padding; only scored rows must describe poses.
    for offset in (0, 7):
        poses(actions[:count, offset:offset + 7])
    predicted = np.stack([integrate(anchors[h], actions[:count, h * 7:h * 7 + 7],
                                    settings.translation_frame) for h in range(2)], axis=1)
    position, rotation = [], []
    for hand in range(2):
        p, r = pose_error(predicted[:, hand], references[:, hand])
        position.append(p)
        rotation.append(r)
    position, rotation = np.array(position).T, np.array(rotation).T
    grip_error = np.abs(actions[:count, 14:16] - grippers)
    predicted_relative = delta_between(predicted[:, 0], predicted[:, 1], "body")
    reference_relative = delta_between(references[:, 0], references[:, 1], "body")
    relative_position, relative_rotation = pose_error(predicted_relative, reference_relative)
    if not all(np.isfinite(values).all() for values in (position, rotation, grip_error, relative_position, relative_rotation)):
        raise ValueError("prediction produces unbounded trajectory errors")
    within = ((position <= settings.position_cm).all(axis=1)
              & (rotation <= settings.rotation_deg).all(axis=1)
              & (grip_error <= settings.gripper_rad).all(axis=1))
    previous = np.concatenate([anchors[None], references[:-1]])
    movement = np.linalg.norm(references[..., :3] - previous[..., :3], axis=-1)
    angular = np.stack([pose_error(previous[:, h], references[:, h])[1] for h in range(2)], axis=1)
    # Motion above 5 mm/s or 2 deg/s excludes stationary-hand noise from this diagnostic.
    moving = ((movement * settings.action_hz > 0.005) | (angular * settings.action_hz > 2)).any(axis=1)
    return {
        "position_cm": position.tolist(), "rotation_deg": rotation.tolist(),
        "gripper_rad": grip_error.tolist(), "inter_hand_position_cm": relative_position.tolist(),
        "inter_hand_rotation_deg": relative_rotation.tolist(), "within_tolerance": within.tolist(),
        "moving": moving.tolist(), "predicted_poses": predicted.tolist(),
        "reference_poses": references.tolist(), "predicted_grippers": actions[:count, 14:16].tolist(),
        "reference_grippers": grippers.tolist(), "actions": actions.tolist(),
        "quaternion_norm_max_error": float(max(np.max(np.abs(np.linalg.norm(actions[:count, offset:offset+4], axis=1) - 1))
                                                for offset in (3, 10))),
    }


def summarize(windows, settings):
    if not windows:
        raise ValueError("episode contains no complete action intervals")
    if any(not 0 < len(w["within_tolerance"]) <= settings.chunk_size
           or len(w["moving"]) != len(w["within_tolerance"]) for w in windows):
        raise ValueError("window masks must cover every valid action interval")
    result = {"windows": len(windows), "timesteps": sum(len(w["within_tolerance"]) for w in windows)}
    for key in ("position_cm", "rotation_deg", "gripper_rad", "inter_hand_position_cm", "inter_hand_rotation_deg"):
        for w in windows:
            shape = (len(w["within_tolerance"]),) + (() if key.startswith("inter_hand_") else (2,))
            if np.shape(w[key]) != shape or np.any(np.asarray(w[key]) < 0):
                raise ValueError("error traces must be non-negative and cover every valid action interval")
        values = np.concatenate([w[key] for w in windows])
        with np.errstate(over="ignore", invalid="ignore"):
            mse = np.mean(np.square(values), axis=0)
        if not np.isfinite(mse).all():
            raise ValueError("prediction produces unbounded squared errors")
        result[key] = {"mean": np.mean(values, axis=0).tolist(), "p95": np.percentile(values, 95, axis=0).tolist(),
                       "mse": mse.tolist(), "rmse": np.sqrt(mse).tolist()}
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized = [np.asarray(result[key]["rmse"]) / scale for key, scale in (
            ("position_cm", settings.position_scale_cm), ("rotation_deg", settings.rotation_scale_deg),
            ("gripper_rad", settings.gripper_scale_rad))]
        result["normalized_mse"] = float(np.mean(np.square(normalized)))
    if not np.isfinite(result["normalized_mse"]):
        raise ValueError("reference scales produce unbounded normalized error")
    result["normalized_rmse"] = float(np.sqrt(result["normalized_mse"]))
    within = np.concatenate([w["within_tolerance"] for w in windows])
    moving = np.concatenate([w["moving"] for w in windows])
    for name, mask in (("score", np.ones(len(within), bool)), ("moving_score", moving), ("stationary_score", ~moving)):
        result[name] = float(100 * within[mask].mean()) if mask.any() else None
    result["latency_ms"] = {"median": float(np.median([w["latency_ms"] for w in windows])),
                            "p95": float(np.percentile([w["latency_ms"] for w in windows], 95))}
    return result
