#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "pyarrow"]
# ///
"""Download the metadata and packed shards needed for selected replay episodes."""

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_arena.replay_suite import Suite


def prepare(episodes, files, fetch):
    info = json.loads(Path(fetch("meta/info.json")).read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("expected LeRobot v3.0")
    fetch("meta/tasks.parquet")
    remaining, selected = set(episodes), []
    columns = ["episode_index", "data/chunk_index", "data/file_index"]
    columns += [f"videos/observation.image.{hand}/{key}" for hand in ("left", "right") for key in ("chunk_index", "file_index")]
    for name in sorted(f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet")):
        rows = pq.read_table(fetch(name), columns=columns, filters=[("episode_index", "in", sorted(remaining))]).to_pylist()
        selected.extend(rows)
        remaining.difference_update(row["episode_index"] for row in rows)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"episodes absent from metadata: {sorted(remaining)}")
    shards = set()
    for row in selected:
        shards.add(info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
        for hand in ("left", "right"):
            key = f"observation.image.{hand}"
            shards.add(info["video_path"].format(video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                                                 file_index=row[f"videos/{key}/file_index"]))
    for name in sorted(shards):
        if name not in files or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"invalid or missing shard: {name}")
        fetch(name)
    return sorted(shards)


def main():
    from huggingface_hub import HfApi, hf_hub_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="airoa-org/yubi-corl2026-umi-arena")
    p.add_argument("--revision", required=True)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episodes", nargs="+", type=int)
    selection.add_argument("--suite", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    suite = Suite(args.suite) if args.suite else None
    if suite:
        args.episodes = list(suite.members)
        suite.validate_gripper_filter()
    if len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0:
        p.error("episodes must be distinct and non-negative")
    api = HfApi()
    revision = api.dataset_info(args.repo, revision=args.revision).sha
    if suite:
        suite.check_revision(revision, args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "replay_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["revision"] != revision or previous["repo"] != args.repo:
            p.error("output already contains a different dataset revision; use another directory")
    manifest.write_text(json.dumps({"repo": args.repo, "revision": revision, "episodes": args.episodes, "status": "preparing"}, indent=2) + "\n")

    def fetch(name):
        print(name, flush=True)
        return hf_hub_download(args.repo, name, repo_type="dataset", revision=revision, local_dir=args.output)

    shards = prepare(args.episodes, api.list_repo_files(args.repo, repo_type="dataset", revision=revision), fetch)
    manifest.write_text(json.dumps({"repo": args.repo, "revision": revision, "episodes": args.episodes,
                                    "shards": shards, "status": "complete"}, indent=2) + "\n")
    print(f"Dataset prepared. Use --dataset-revision {revision}")


if __name__ == "__main__":
    main()
