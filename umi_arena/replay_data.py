"""Read selected episodes from a local LeRobot v3 dataset without loading a model stack."""

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from umi_arena.replay_metrics import delta_between, integrate, pose_error, poses

HANDS = ("left", "right")
INTER_HAND = "observation.pose.left_hand_root_to_right_hand_root.absolute"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("replay requires a LeRobot v3.0 dataset")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("dataset fps must be finite and positive")
        self.files = {self.root / "meta/info.json", self.root / "meta/tasks.parquet"}
        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pylist()
        # The published practice dataset stores task text in the pandas index column.
        self.tasks = {int(row["task_index"]): row.get("task", row.get("__index_level_0__")) for row in tasks}

    def path(self, pattern, **values):
        path = (self.root / pattern.format(**values)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("dataset path escapes its root")
        self.files.add(path)
        return path

    def episode(self, index):
        columns = ["episode_index", "length", "data/chunk_index", "data/file_index", "task_success",
                   "uuid", "short_horizon_task", "primitive_action", "success_short_horizon_task", "episode_id"]
        for hand in HANDS:
            columns.extend(f"videos/observation.image.{hand}/{key}" for key in
                           ("chunk_index", "file_index", "from_timestamp", "to_timestamp"))
        for path in sorted(self.root.glob("meta/episodes/chunk-*/file-*.parquet")):
            rows = pq.read_table(path, columns=columns, filters=[("episode_index", "=", index)]).to_pylist()
            if rows:
                self.files.add(path)
                if len(rows) != 1:
                    raise ValueError(f"duplicate metadata for episode {index}")
                return Episode(self, rows[0])
        raise ValueError(f"episode {index} is absent from the local metadata")

    def hashes(self):
        return {str(p.relative_to(self.root)): sha256(p) for p in sorted(self.files) if p.is_file()}


class Episode:
    def __init__(self, dataset, metadata):
        self.dataset, self.metadata = dataset, metadata
        if metadata["task_success"] is not True:
            raise ValueError("reference episode must have task_success=true")
        path = dataset.path(dataset.info["data_path"], chunk_index=metadata["data/chunk_index"],
                            file_index=metadata["data/file_index"])
        columns = ["timestamp", "frame_index", "task_index", "observation.joint_states", "action.joint_states", INTER_HAND]
        columns += [f"observation.pose.{h}_hand_root.absolute" for h in HANDS]
        columns += [f"action.pose.{h}_hand_root.relative" for h in HANDS]
        self.data = {k: np.asarray(v) for k, v in pq.read_table(
            path, columns=columns, filters=[("episode_index", "=", metadata["episode_index"])]).to_pydict().items()}
        self.times = self.data["timestamp"].reshape(-1)
        self.length = len(self.times)
        if self.length != metadata["length"] or self.length < 3:
            raise ValueError("episode length differs from metadata or contains fewer than three frames")
        if not np.array_equal(self.data["frame_index"].reshape(-1), np.arange(self.length)):
            raise ValueError("episode frame indices must be ordered and contiguous")
        if not np.isfinite(self.times).all() or not np.allclose(np.diff(self.times), 1 / dataset.fps, atol=1e-5):
            raise ValueError("episode timestamps do not match the dataset frame rate")
        self.hand_poses = np.stack([poses(self.data[f"observation.pose.{h}_hand_root.absolute"]) for h in HANDS], axis=1)
        self.joints = np.asarray(self.data["observation.joint_states"], dtype=float)
        self.joint_actions = np.asarray(self.data["action.joint_states"], dtype=float)
        if self.joints.shape != (self.length, 2) or not np.isfinite(self.joints).all() or not np.isfinite(self.joint_actions).all():
            raise ValueError("invalid gripper references")
        poses(self.data[INTER_HAND])  # Reject invalid inter-hand poses before sending observations to a policy.

    def audit(self):
        errors = {}
        for frame in ("body", "world"):
            position, rotation = [], []
            for hand, name in enumerate(HANDS):
                recorded = self.data[f"action.pose.{name}_hand_root.relative"][1:]
                predicted = integrate(self.hand_poses[0, hand], recorded, frame)
                p, r = pose_error(predicted, self.hand_poses[1:, hand])
                position.append(float(p.max()))
                rotation.append(float(r.max()))
            errors[frame] = {"position_max_cm": position, "rotation_max_deg": rotation}
        matching = [frame for frame, error in errors.items()
                    if max(error["position_max_cm"]) < 0.01 and max(error["rotation_max_deg"]) < 0.01]
        joint_error = float(np.max(np.abs(self.joint_actions[:-1] - self.joints[1:])))
        if not matching or joint_error > 1e-6:
            raise ValueError(f"reference reconstruction failed: {errors}; gripper shift error={joint_error}")
        return {"translation_frames_matching": matching, "pose_timing": "previous",
                "gripper_timing": "next", "gripper_max_error_rad": joint_error, "reconstruction": errors}

    def windows(self, settings):
        stride = settings.stride(self.dataset.fps)
        first = stride if settings.pose_timing == "previous" else 0
        for start in range(first, self.length - stride, settings.chunk_size * stride):
            count = min(settings.chunk_size, (self.length - 1 - start) // stride)
            pose_start = start - stride if settings.pose_timing == "previous" else start
            pose_indices = pose_start + np.arange(1, count + 1) * stride
            joint_indices = start + np.arange(1, count + 1) * stride
            yield start, self.hand_poses[pose_start], self.hand_poses[pose_indices], self.joints[joint_indices]

    def observation(self, index):
        prompt = self.dataset.tasks.get(int(self.data["task_index"][index]))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"missing instruction at frame {index}")
        observation = {INTER_HAND: self.data[INTER_HAND][index].astype(np.float32),
                       "observation.joint_states": self.joints[index].astype(np.float32), "prompt": prompt}
        for hand in HANDS:
            key = f"observation.image.{hand}"
            prefix = f"videos/{key}/"
            path = self.dataset.path(self.dataset.info["video_path"], video_key=key,
                                     chunk_index=self.metadata[prefix + "chunk_index"],
                                     file_index=self.metadata[prefix + "file_index"])
            target = self.metadata[prefix + "from_timestamp"] + self.times[index] - self.times[0]
            if target >= self.metadata[prefix + "to_timestamp"]:
                raise ValueError("video timestamp extends past the episode")
            observation[key] = video_frame(path, target, self.dataset.fps)
        return observation


def video_frame(path, timestamp, fps):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is None or frame.time < timestamp - 0.5 / fps:
                continue
            if abs(frame.time - timestamp) > 0.5 / fps:
                break
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (480, 640, 3):
                raise ValueError(f"expected 480x640 RGB image, got {image.shape}")
            return image
    raise ValueError(f"no video frame within half a timestep of {timestamp} in {path}")


def reference_actions(anchors, references, joints, settings):
    previous = np.concatenate([anchors[None], references[:-1]])
    result = np.concatenate([delta_between(previous[:, h], references[:, h], settings.translation_frame)
                             for h in range(2)] + [joints], axis=1)
    return np.concatenate([result, np.repeat(result[-1:], settings.chunk_size - len(result), axis=0)])
