#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy", "scipy>=1.11", "pyarrow", "av", "pillow", "websockets>=13", "huggingface_hub",
#   "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client",
# ]
# ///
"""Compare a served policy with recorded LeRobot v3 episodes."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import check
from umi_arena.replay_data import Dataset, reference_actions, sha256
from umi_arena.replay_metrics import Settings, compare, summarize
from umi_arena.replay_report import write_episode, write_metadata, write_report, thumbnail
from umi_arena.replay_suite import Suite, aggregate_suite, averaged
from umi_arena.replay_grippers import validate_raw


def fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and not {"__pycache__", ".cache", ".git"}.intersection(relative.parts):
            digest.update(str(relative).encode() + b"\0" + sha256(path).encode())
    return digest.hexdigest()


def cleanup_container(name, log):
    warnings, removed = [], False
    try:
        check._save_logs(name, log)
    except Exception as exc:
        warnings.append(f"could not save logs for {name}: {exc}")
    finally:
        try:
            result = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30)
            removed = result.returncode == 0
            if not removed:
                warnings.append(f"could not remove {name}: {result.stderr.strip() or result.returncode}")
        except (OSError, subprocess.SubprocessError) as exc:
            warnings.append(f"could not remove {name}: {exc}")
    if not removed:
        try:
            result = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
                                    capture_output=True, text=True, timeout=30)
            removed = result.returncode == 0 and name not in result.stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            pass
        if not removed:
            warnings.append(f"cannot confirm {name} was removed; check Docker before starting another recording")
    return warnings, removed


def evaluate_episode(episode, settings, infer=None, baseline=None):
    windows = []
    for start, anchors, reference, grippers in episode.windows(settings):
        observation = episode.observation(start) if infer else None
        if infer:
            response, latency = infer(observation)
            if not isinstance(response, dict) or "error" in response or "actions" not in response:
                raise ValueError(f"invalid server response: {str(response)[:300]}")
            actions = response["actions"]
        elif baseline == "reference":
            actions, latency = reference_actions(anchors, reference, grippers, settings), 0.0
        elif baseline == "hold":
            actions = np.zeros((settings.chunk_size, 16))
            actions[:, [6, 13]] = 1
            actions[:, 14:16] = episode.joints[start]
            latency = 0.0
        else:
            raise ValueError("select a server or a baseline")
        window = compare(actions, anchors, reference, grippers, settings)
        window.update(frame=start, timestamp=float(episode.times[start]), latency_ms=float(latency),
                      prompt=episode.dataset.tasks[int(episode.data["task_index"][start])])
        if observation:
            window["images"] = [thumbnail(observation[f"observation.image.{h}"]) for h in ("left", "right")]
        window["summary"] = summarize([window], settings)
        windows.append(window)
    return {"summary": summarize(windows, settings), "windows": windows}


def aggregate(episodes):
    if not episodes or any(e["status"] != "complete" for e in episodes):
        return None  # A failed episode must not improve a submission's score by disappearing.
    tasks = {}
    for episode in episodes:
        tasks.setdefault(episode["task"], {}).setdefault(episode["episode_index"], []).append(episode["summary"])
    summaries = {task: averaged([averaged(repeats) for repeats in values.values()]) for task, values in tasks.items()}
    return {**averaged(list(summaries.values())), "task_scores": {t: s["score"] for t, s in summaries.items()},
            "task_errors": {t: s["normalized_rmse"] for t, s in summaries.items()},
            "weighting": "mean repeats, then episodes within each task, then equal tasks"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, help="local LeRobot v3 root; only selected shards are needed")
    p.add_argument("--dataset-revision", help="dataset commit SHA or local version label; files are also hashed")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--episodes", nargs="+", type=int)
    selection.add_argument("--suite", type=Path, help="fixed task steps and recording membership")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--submission", type=Path, help="fresh runner per episode or suite recording, and per repeat")
    source.add_argument("--server", help="fresh WebSocket server; one episode and one repeat per invocation")
    source.add_argument("--baseline", choices=("reference", "hold"))
    p.add_argument("--checkpoint-id", help="model revision or training run identifier")
    p.add_argument("--action-hz", type=float, help="time interval represented by each model output row")
    p.add_argument("--translation-frame", choices=("body", "world"))
    p.add_argument("--pose-timing", choices=("previous", "next"),
                   help="previous matches stored training labels; next compares future motion from the current observation")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    p.add_argument("--image", help="runner image tag or digest; required with --submission")
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--hf-cache", type=Path)
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--load-timeout", type=float, default=300)
    p.add_argument("--request-timeout", type=float, default=30)
    return p


def main(argv=None):
    p = parser()
    argv = sys.argv[1:] if argv is None else argv
    args = p.parse_args(argv)
    if not all((args.dataset, args.dataset_revision, args.action_hz, args.translation_frame, args.pose_timing,
                args.episodes or args.suite, args.submission or args.server or args.baseline)):
        p.error("evaluation requires dataset, dataset revision, timing, episode selection and a prediction source")
    suite, rule = None, None
    if args.suite:
        try:
            suite = Suite(args.suite)
            suite.check_revision(args.dataset_revision)
            rule = suite.validate_gripper_filter()
        except (ValueError, KeyError, TypeError, OSError) as exc:
            p.error(str(exc))
        args.episodes = list(suite.members)
    if args.repeats < 1 or len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0:
        p.error("use distinct non-negative episodes and at least one repeat")
    if args.server and (len(args.episodes) != 1 or args.repeats != 1):
        p.error("--server requires one episode and one repeat; --submission isolates multiple episodes")
    if not args.baseline and not args.checkpoint_id:
        p.error("--checkpoint-id is required for a policy")
    if args.submission and not args.image:
        p.error("--image is required with --submission")
    if any(not np.isfinite(t) or t <= 0 for t in (args.load_timeout, args.request_timeout)):
        p.error("timeouts must be finite and positive")
    try:
        settings = Settings(args.action_hz, args.translation_frame, args.pose_timing)
    except ValueError as exc:
        p.error(str(exc))
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        p.error(f"output already exists: {args.output}; choose a new --output directory")
    except OSError as exc:
        p.error(f"cannot create output directory {args.output}: {exc}")
    report = {"version": 3, "status": "running", "scope": "demonstration agreement; tolerances are diagnostic",
              "created_at": datetime.now(timezone.utc).isoformat(), "settings": asdict(settings),
              "dataset_revision": args.dataset_revision, "requested_episodes": args.episodes,
              "repeats": args.repeats, "checkpoint_id": args.checkpoint_id, "baseline": args.baseline,
              "episodes": [], "aggregate": None}
    if suite:
        report.update(suite=suite.spec, suite_sha256=suite.sha256, coverage=suite.coverage(),
                      server_reset="fresh process per recording and repeat; shared across its task steps")
    dataset, cleanup_safe = None, True
    gripper_checks = set()
    try:
        if rule:
            report["gripper_check"] = {"minimum_rad": rule["minimum_rad"],
                                       "scope": "every selected episode", "status": "pending"}
        dataset = Dataset(args.dataset)
        manifest = dataset.root / "replay_manifest.json"
        if manifest.is_file():
            prepared = json.loads(manifest.read_text())
            if prepared.get("status") != "complete" or prepared.get("revision") != args.dataset_revision:
                raise ValueError("dataset revision differs from the prepared manifest or preparation is incomplete")
            dataset.files.add(manifest)
            if suite:
                suite.check_revision(prepared["revision"], prepared.get("repo"))
        settings.stride(dataset.fps)
        report.update(dataset_fps=dataset.fps, dataset_root=str(dataset.root))
        report["source_hashes"] = {str(path.relative_to(Path(__file__).parent)): sha256(path) for path in
                                  [Path(__file__), Path(check.__file__),
                                   *sorted((Path(__file__).parent / "umi_arena").glob("replay_*.py"))]}
        if args.submission:
            args.submission = args.submission.resolve()
            if not (args.submission / "policy.py").is_file():
                raise ValueError("submission must contain policy.py and the checkpoint")
            args.volume = []
            check.preflight(args, check.Report())
            print("Hashing the submission for reproducibility...", flush=True)
            report.update(submission_sha256=fingerprint(args.submission), image=check.image_digest(args.image))
        write_metadata(args.output, report)
        groups = suite.groups if suite else [[index] for index in args.episodes]
        for repeat in range(args.repeats):
            for indices in groups:
                if not cleanup_safe:
                    raise RuntimeError("previous policy container was not confirmed removed; stopping before the next recording")
                entries = [{"episode_index": index, "repeat": repeat, "status": "failed"} for index in indices]
                report["episodes"].extend(entries)
                client, runner_report = None, check.Report()
                log = args.output / f"episode-{indices[0]}-repeat-{repeat}.log"
                try:
                    episodes = [dataset.episode(index) for index in indices]
                    for episode, entry in zip(episodes, entries):
                        task = episode.metadata["short_horizon_task"]
                        entry.update(uuid=episode.metadata["uuid"], task=task if isinstance(task, str) else " / ".join(task))
                        if suite:
                            entry.update(suite.validate(episode))
                            if rule:
                                validate_raw(episode, rule["minimum_rad"])
                                gripper_checks.add((entry["episode_index"], repeat))
                        entry["audit"] = episode.audit()
                    url = args.server
                    if args.submission:
                        check.start_container(args, args.submission, log, runner_report)
                        url = f"ws://127.0.0.1:{args.port}"
                    if url:
                        client = check.Client(url, keepalive_s=20)
                        metadata = client.metadata(10)
                        if not isinstance(metadata, dict) or metadata.get("action_dim") != 16:
                            raise ValueError(f"invalid runner metadata: {metadata}")
                        for entry in entries:
                            entry["server_metadata"] = metadata
                    for episode, entry in zip(episodes, entries):
                        entry.update(evaluate_episode(episode, settings,
                                     (lambda obs: client.infer(obs, timeout_s=args.request_timeout)) if client else None,
                                     args.baseline))
                        write_episode(args.output, {**entry, "status": "complete"})
                        entry["status"] = "complete"
                        print(f"episode {entry['episode_index']}, repeat {repeat}: normalized RMSE {entry['summary']['normalized_rmse']:.3f}", flush=True)
                except Exception as exc:
                    for entry in entries:
                        if entry["status"] != "complete":
                            entry["error"] = f"{type(exc).__name__}: {exc}"
                            print(f"episode {entry['episode_index']}, repeat {repeat}: {entry['error']}", file=sys.stderr, flush=True)
                finally:
                    try:
                        if client:
                            client.close()
                    finally:
                        if name := runner_report.info.get("container"):
                            warnings, cleanup_safe = cleanup_container(name, log)
                            if warnings:
                                report.setdefault("warnings", []).extend(warnings)
                                for warning in warnings:
                                    print(f"warning: {warning}", file=sys.stderr)
        report["aggregate"] = aggregate_suite(report["episodes"], suite, args.repeats) if suite else aggregate(report["episodes"])
        report["status"] = "complete" if report["aggregate"] is not None else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="evaluation interrupted")
    finally:
        if dataset:
            try:
                report["dataset_files_sha256"] = dataset.hashes()
            except Exception as exc:
                error = "\n".join(filter(None, [report.get("error"), f"dataset hashing failed: {exc}"]))
                report.update(status="failed", aggregate=None, error=error)
        if rule:
            expected = len(args.episodes) * args.repeats
            report["gripper_check"].update(status="passed" if len(gripper_checks) == expected else "incomplete",
                                           checked_episodes=len(gripper_checks), expected_episodes=expected)
        for entry in report["episodes"]:
            if entry["status"] != "complete":
                write_episode(args.output, entry)
        write_metadata(args.output, report)
        write_report(args.output, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
