"""Fixed task steps and recording membership for replay comparisons."""

import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean

from umi_arena.replay_grippers import cutoff


def normalized(text):
    return " ".join(text.strip().rstrip(".").casefold().split())


class Suite:
    def __init__(self, path):
        content = json.dumps(path).encode() if isinstance(path, dict) else Path(path).read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        self.spec = json.loads(content)
        self.members = {}
        self.groups = []
        task_ids, uuids = set(), set()
        if self.spec.get("version") != 1 or not self.spec.get("tasks"):
            raise ValueError("suite must have version 1 and at least one task")
        for task in self.spec["tasks"]:
            if task["id"] in task_ids or not task["steps"] or not task["recordings"]:
                raise ValueError("suite tasks need distinct IDs, steps and recordings")
            task_ids.add(task["id"])
            for recording in task["recordings"]:
                indices = recording["episodes"]
                if recording["uuid"] in uuids or len(indices) != len(task["steps"]):
                    raise ValueError("each recording must have a distinct UUID and every task step")
                uuids.add(recording["uuid"])
                self.groups.append(indices)
                for step, (index, instruction) in enumerate(zip(indices, task["steps"]), 1):
                    if type(index) is not int or index < 0 or index in self.members:
                        raise ValueError("suite episode indices must be distinct non-negative integers")
                    if not isinstance(instruction, str) or not instruction.strip():
                        raise ValueError("suite steps need nonempty instructions")
                    self.members[index] = {"task": task["name"], "open_task_id": task["id"],
                                           "dataset_task": task["dataset_task"], "step": step,
                                           "subtask": instruction, "uuid": recording["uuid"]}
        uncovered = self.spec["uncovered_tasks"]
        all_ids = [t["id"] for t in self.spec["tasks"] + uncovered]
        if len(set(all_ids)) != len(all_ids) or len(all_ids) != self.spec["total_open_tasks"]:
            raise ValueError("suite must list every selected and uncovered open task exactly once")

    def validate_gripper_filter(self):
        rule = self.spec.get("gripper_filter")
        if rule is not None:
            if not isinstance(rule, dict) or "minimum_rad" not in rule or set(rule) - {"minimum_rad", "scope"}:
                raise ValueError("replay gripper_filter supports only minimum_rad and optional scope")
            cutoff(rule["minimum_rad"])
        return rule

    def check_revision(self, revision, repo=None):
        if revision != self.spec["dataset_revision"] or (repo and repo != self.spec["dataset_repo"]):
            raise ValueError("suite dataset revision or repository does not match")

    def validate(self, episode):
        meta = episode.metadata
        member = self.members[meta["episode_index"]]
        if (meta["uuid"] != member["uuid"] or meta["short_horizon_task"] != [member["dataset_task"]]
                or meta.get("success_short_horizon_task") is not True
                or meta.get("episode_id") != f'{member["uuid"]}:{member["step"] - 1}'):
            raise ValueError(f"episode {meta['episode_index']} does not match its suite recording and step")
        expected = normalized(member["subtask"])
        recorded = meta.get("primitive_action", [])
        prompts = {episode.dataset.tasks.get(int(i), "") for i in episode.data["task_index"].reshape(-1)}
        if len(recorded) != 1 or normalized(recorded[0]) != expected or {normalized(p) for p in prompts} != {expected}:
            raise ValueError(f"episode {meta['episode_index']} instruction differs from the suite step")
        return dict(member)

    def coverage(self):
        return {"selected_tasks": [{"id": t["id"], "name": t["name"]} for t in self.spec["tasks"]],
                "total_open_tasks": self.spec["total_open_tasks"],
                "uncovered_tasks": self.spec["uncovered_tasks"],
                "selected_steps": sum(len(t["steps"]) for t in self.spec["tasks"])}


def averaged(summaries):
    result = {"score": mean(s["score"] for s in summaries),
              "normalized_mse": mean(s["normalized_mse"] for s in summaries)}
    result["normalized_rmse"] = sqrt(result["normalized_mse"])
    for name in ("moving_score", "stationary_score"):
        values = [s[name] for s in summaries if s.get(name) is not None]  # a selection can hold no moving timestep
        result[name] = mean(values) if values else None
    for key in ("position_cm", "rotation_deg", "gripper_rad"):
        result[key] = {stat: [mean(s[key][stat][hand] for s in summaries) for hand in range(2)]
                       for stat in ("mean", "mse")}
        result[key]["rmse"] = [sqrt(v) for v in result[key]["mse"]]
    return result


def aggregate_suite(entries, suite, repeats):
    expected = {(index, repeat) for index in suite.members for repeat in range(repeats)}
    actual = [(e["episode_index"], e["repeat"]) for e in entries]
    if (set(actual) != expected or len(actual) != len(expected)
            or any(e["status"] != "complete" for e in entries)):
        return None
    by_episode = {index: averaged([e["summary"] for e in entries if e["episode_index"] == index])
                  for index in suite.members}
    tasks = []
    for task in suite.spec["tasks"]:
        steps = []
        for step, instruction in enumerate(task["steps"], 1):
            indices = [r["episodes"][step - 1] for r in task["recordings"]]
            steps.append({"step": step, "instruction": instruction, "episodes": indices,
                          "summary": averaged([by_episode[index] for index in indices])})
        tasks.append({"id": task["id"], "name": task["name"], "recordings": len(task["recordings"]),
                      "steps": steps, "summary": averaged([s["summary"] for s in steps]),
                      "weakest_step": max(steps, key=lambda s: s["summary"]["normalized_mse"])["step"]})
    return {**averaged([t["summary"] for t in tasks]), "tasks": tasks,
            "task_scores": {t["name"]: t["summary"]["score"] for t in tasks},
            "task_errors": {t["name"]: t["summary"]["normalized_rmse"] for t in tasks},
            "weighting": "valid timesteps within each PA, then equal repeats, recordings per step, steps per task, and selected tasks"}
