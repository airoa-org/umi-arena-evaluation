"""Reference alignment and scoring must survive resampling, chunk tails, and protocol failures."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("scipy")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
av = pytest.importorskip("av")
from scipy.spatial.transform import Rotation

import replay
from umi_arena.replay_data import Dataset, INTER_HAND, reference_actions, video_frame
from umi_arena.replay_metrics import Settings, compare, delta_between, integrate, pose_error, summarize
from umi_arena.replay_report import write_episode, write_report
from umi_arena.replay_suite import Suite, aggregate_suite


def metric_summary(score):
    error = 100 - score
    return {"score": score, "normalized_mse": error ** 2, "normalized_rmse": error,
            **{k: {"mean": [error] * 2, "mse": [error ** 2] * 2, "rmse": [error] * 2}
               for k in ("position_cm", "rotation_deg", "gripper_rad")}}


def test_rmse_preserves_units_and_penalizes_larger_misses():
    settings = Settings(30, "body", "previous")
    window = {"position_cm": [[3, 0], [4, 0]], "rotation_deg": [[0, 0], [0, 10]],
              "gripper_rad": [[0, 0.1], [0, 0.1]], "inter_hand_position_cm": [0, 0],
              "inter_hand_rotation_deg": [0, 0], "within_tolerance": [False, False],
              "moving": [True, True], "latency_ms": 1}
    result = summarize([window], settings)
    assert result["position_cm"]["rmse"] == pytest.approx([np.sqrt(12.5), 0])
    assert result["rotation_deg"]["rmse"] == pytest.approx([0, np.sqrt(50)])
    assert result["gripper_rad"]["rmse"] == pytest.approx([0, 0.1])
    assert result["normalized_mse"] == pytest.approx((12.5 / 4 + 50 / 25 + 0.01 / 0.0025) / 6)
    relaxed = summarize([window], Settings(30, "body", "previous", position_cm=100))
    assert relaxed["normalized_rmse"] == result["normalized_rmse"]
    for key in ("position_cm", "rotation_deg", "gripper_rad"):
        window[key] = (np.asarray(window[key]) * 2).tolist()
    worse = summarize([window], settings)
    assert worse["score"] == result["score"] == 0
    assert worse["normalized_rmse"] == pytest.approx(result["normalized_rmse"] * 2)
    scaled = summarize([window], Settings(30, "body", "previous", position_scale_cm=4,
                                         rotation_scale_deg=10, gripper_scale_rad=0.1))
    assert scaled["normalized_rmse"] == pytest.approx(result["normalized_rmse"])
    window["position_cm"][0][0] = 1e308
    with pytest.raises(ValueError, match="unbounded squared"):
        summarize([window], settings)


@pytest.mark.parametrize("key", ["position_scale_cm", "rotation_scale_deg", "gripper_scale_rad"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_reference_scales(key, value):
    with pytest.raises(ValueError, match="finite and positive"):
        Settings(30, "body", "previous", **{key: value})


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "__index_level_0__": "move the cup"}]), root / "meta/tasks.parquet")
    rows, metadata = [], []
    for episode in range(2):
        t = np.arange(50) / 30
        rotations = Rotation.from_euler("z", (0.7 + t * 0.5)[:, None]).as_quat()
        left = np.column_stack([t * .1, .03 * np.sin(t * 3), np.ones(50) * .2, rotations])
        right = left.copy()
        right[:, 0] += .25
        relative = delta_between(left, right, "body")
        left_delta = np.vstack([[0, 0, 0, 0, 0, 0, 1], delta_between(left[:-1], left[1:], "body")])
        right_delta = np.vstack([[0, 0, 0, 0, 0, 0, 1], delta_between(right[:-1], right[1:], "body")])
        joints = np.column_stack([.2 + t * .1, .4 - t * .1])
        for i in range(50):
            rows.append({"episode_index": episode, "frame_index": i, "timestamp": t[i], "task_index": 0,
                         "observation.joint_states": joints[i].tolist(), "action.joint_states": joints[min(i+1,49)].tolist(),
                         INTER_HAND: relative[i].tolist(), "observation.pose.left_hand_root.absolute": left[i].tolist(),
                         "observation.pose.right_hand_root.absolute": right[i].tolist(),
                         "action.pose.left_hand_root.relative": left_delta[i].tolist(),
                         "action.pose.right_hand_root.relative": right_delta[i].tolist()})
        meta = {"episode_index": episode, "length": 50, "data/chunk_index": 0, "data/file_index": 0,
                "task_success": True, "uuid": f"group-{episode}", "short_horizon_task": ["cups"],
                "primitive_action": ["move the cup"], "success_short_horizon_task": True,
                "episode_id": f"group-{episode}:0"}
        for hand in ("left", "right"):
            meta.update({f"videos/observation.image.{hand}/{k}": v for k,v in
                         {"chunk_index": 0, "file_index": 0, "from_timestamp": episode * 50/30,
                          "to_timestamp": (episode+1) * 50/30}.items()})
        metadata.append(meta)
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


@pytest.mark.parametrize("frame", ["world", "body"])
def test_noncommuting_pose_composition(frame):
    before = np.r_[1, 2, 3, Rotation.from_euler("x", 1).as_quat()]
    after = np.r_[2, 4, 1, Rotation.from_euler("zy", [.7, .8]).as_quat()]
    delta = delta_between(before, after, frame)
    p, r = pose_error(integrate(before, delta[None], frame), after[None])
    assert p[0] < 1e-10
    assert r[0] < 1e-10
    opposite = after.copy()
    opposite[3:] *= -1
    assert pose_error(after[None], opposite[None])[1][0] == pytest.approx(0)


@pytest.mark.parametrize("hz,timing", [(30,"previous"), (30,"next"), (10,"previous"), (10,"next")])
def test_reference_scores_perfectly_with_resampling_and_tail(dataset_root, hz, timing):
    episode = Dataset(dataset_root).episode(0)
    settings = Settings(hz, "body", timing)
    result = replay.evaluate_episode(episode, settings, baseline="reference")
    assert episode.audit()["translation_frames_matching"] == ["body"]
    assert result["summary"]["score"] == 100
    assert result["summary"]["normalized_rmse"] == pytest.approx(0, abs=1e-10)
    assert max(result["summary"]["position_cm"]["mean"]) < 1e-10
    stride = settings.stride(30)
    expected = (episode.length - 1 - (stride if timing == "previous" else 0)) // stride
    assert result["summary"]["timesteps"] == expected
    assert len(result["windows"][-1]["within_tolerance"]) == (expected % 16 or 16)
    assert all(len(w["actions"]) == 16 for w in result["windows"])
    assert all(len(w["reference_poses"]) <= 16 for w in result["windows"])


def test_resampling_composes_all_three_steps(dataset_root):
    episode = Dataset(dataset_root).episode(1)
    settings = Settings(10, "body", "next")
    start, anchors, ref, joints = next(episode.windows(settings))
    action = reference_actions(anchors, ref, joints, settings)
    assert start == 0
    np.testing.assert_allclose(action[0,:7], delta_between(episode.hand_poses[0,0], episode.hand_poses[3,0], "body"))
    assert not np.allclose(action[0,:3], episode.data["action.pose.left_hand_root.relative"][3,:3])
    np.testing.assert_allclose(action[0,14:], episode.joints[3])


def test_hold_policy_is_worse_than_reference(dataset_root):
    episode = Dataset(dataset_root).episode(0)
    result = replay.evaluate_episode(episode, Settings(30, "body", "next"), baseline="hold")
    assert result["summary"]["score"] < 100
    assert result["summary"]["moving_score"] < 100


@pytest.mark.parametrize("defect", ["nan", "zero_quaternion", "short", "wrong_width", "overflow"])
def test_invalid_prediction_cannot_receive_score(dataset_root, defect):
    episode = Dataset(dataset_root).episode(0)
    settings = Settings(30, "body", "next")
    _, anchor, ref, joints = next(episode.windows(settings))
    actions = reference_actions(anchor, ref, joints, settings)
    if defect == "nan": actions[0,0] = np.nan
    if defect == "zero_quaternion": actions[0,3:7] = 0
    if defect == "short": actions = actions[:8]
    if defect == "wrong_width": actions = actions[:,:15]
    if defect == "overflow": actions[:,0] = 1e300
    with pytest.raises(ValueError):
        compare(actions, anchor, ref, joints, settings)


@pytest.mark.parametrize("count", [3, 16])
def test_unused_action_padding_does_not_require_valid_quaternions(dataset_root, count):
    episode = Dataset(dataset_root).episode(0)
    settings = Settings(30, "body", "next")
    _, anchor, ref, joints = next(episode.windows(settings))
    actions = np.vstack([reference_actions(anchor, ref, joints, settings), np.zeros((16, 16))])
    actions[count:] = 0
    result = compare(actions, anchor, ref[:count], joints[:count], settings)
    assert all(result["within_tolerance"])
    assert result["quaternion_norm_max_error"] < 1e-10
    actions[-1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite actions"):
        compare(actions, anchor, ref[:count], joints[:count], settings)
    actions[-1, 0] = 0
    actions[count - 1, 3:7] = 0
    with pytest.raises(ValueError, match="quaternion"):
        compare(actions, anchor, ref[:count], joints[:count], settings)


def test_reference_audit_rejects_shifted_pose_labels(dataset_root):
    episode = Dataset(dataset_root).episode(0)
    key = "action.pose.left_hand_root.relative"
    episode.data[key] = np.roll(episode.data[key], -1, axis=0)
    with pytest.raises(ValueError, match="reconstruction"):
        episode.audit()


def test_failed_episode_suppresses_aggregate_and_repeats_do_not_reweight_tasks():
    entries = [{"status":"complete", "task":"a", "episode_index":i, "summary": metric_summary(s)}
               for i,s in [(0,100),(0,0),(1,100)]]
    entries.append({"status":"complete", "task":"b", "episode_index":2, "summary": metric_summary(0)})
    assert replay.aggregate(entries)["score"] == 37.5
    assert replay.aggregate(entries)["normalized_rmse"] == pytest.approx(np.sqrt(6250))
    entries.append({"status":"failed"})
    assert replay.aggregate(entries) is None


def test_cli_writes_reports_and_hashes(dataset_root, tmp_path):
    output = tmp_path / "report"
    code = replay.main(["--dataset",str(dataset_root),"--dataset-revision","test","--episodes","0","1",
                        "--baseline","reference","--action-hz","30","--translation-frame","body",
                        "--pose-timing","previous","--output",str(output)])
    assert code == 0
    report = json.loads((output / "report.json").read_text())
    assert report["aggregate"]["score"] == 100
    assert len(report["dataset_files_sha256"]) == 4
    assert "const report=" in (output / "report.html").read_text()


def test_missing_episode_is_reported(dataset_root, tmp_path):
    output = tmp_path / "failed"
    assert replay.main(["--dataset",str(dataset_root),"--dataset-revision","test","--episodes","99",
                        "--baseline","hold","--action-hz","30","--translation-frame","body",
                        "--pose-timing","next","--output",str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["aggregate"] is None and report["episodes"][0]["status"] == "failed"


def test_html_escapes_dataset_text(tmp_path):
    write_report(tmp_path, {"prompt":"</script><script>alert('oops')</script>"})
    assert "</script><script>alert" not in (tmp_path / "report.html").read_text()


@pytest.mark.parametrize("repeats", [1, 3])
def test_episode_checkpoints_are_written_once_and_final_reports_match(dataset_root, suite_path, tmp_path, monkeypatch, repeats):
    from umi_arena.replay_data import Episode

    output = tmp_path / "incremental"
    writes = []
    audits = []
    original_audit = Episode.audit
    def audit(episode):
        audits.append(episode.metadata["episode_index"])
        return original_audit(episode)
    monkeypatch.setattr(Episode, "audit", audit)
    original_write = Path.write_text
    def record_write(path, data, *args, **kwargs):
        if output in path.parents:
            writes.append(path.relative_to(output))
        return original_write(path, data, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", record_write)
    assert replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--suite", str(suite_path),
                        "--baseline", "reference", "--action-hz", "30", "--translation-frame", "body",
                        "--pose-timing", "previous", "--repeats", str(repeats), "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert len(report["episodes"]) == 2 * repeats
    assert audits == [0, 1] * repeats
    for entry in report["episodes"]:
        path = Path("episodes") / f"episode-{entry['episode_index']}-repeat-{entry['repeat']}.json"
        assert json.loads((output / path).read_text()) == entry
        assert writes.count(path.with_suffix(".json.tmp")) == 1
    assert writes.count(Path("report.json.tmp")) == writes.count(Path("report.html.tmp")) == 1
    assert json.loads((output / "metadata.json").read_text()) == {k: v for k, v in report.items() if k != "episodes"}
    html = (output / "report.html").read_text()
    embedded = html.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
    assert json.loads(embedded) == report


@pytest.mark.parametrize("failure", [KeyboardInterrupt, ValueError])
def test_interrupted_or_failed_run_keeps_completed_episodes(dataset_root, suite_path, tmp_path, monkeypatch, failure):
    output = tmp_path / "interrupted"
    evaluate = replay.evaluate_episode
    calls = []
    def stop_on_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            checkpoint = json.loads((output / "episodes/episode-0-repeat-0.json").read_text())
            assert checkpoint["status"] == "complete" and checkpoint["windows"]
            assert json.loads((output / "metadata.json").read_text())["status"] == "running"
            assert not (output / "report.json").exists()
            assert not (output / "report.html").exists()
            raise failure("stopped during second episode")
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(replay, "evaluate_episode", stop_on_second)
    assert replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--suite", str(suite_path),
                        "--baseline", "reference", "--action-hz", "30", "--translation-frame", "body",
                        "--pose-timing", "previous", "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
    assert report["aggregate"] is None
    assert [e["status"] for e in report["episodes"]] == ["complete", "failed"]
    for entry in report["episodes"]:
        assert json.loads((output / f"episodes/episode-{entry['episode_index']}-repeat-0.json").read_text()) == entry
    assert (output / "report.html").is_file()


def test_abrupt_exit_preserves_completed_episode(dataset_root, tmp_path):
    output = tmp_path / "killed"
    code = """
import os, replay, sys
evaluate = replay.evaluate_episode
calls = 0
def stop_on_second(*args, **kwargs):
    global calls
    calls += 1
    if calls == 2:
        os._exit(23)
    return evaluate(*args, **kwargs)
replay.evaluate_episode = stop_on_second
replay.main(sys.argv[1:])
"""
    result = subprocess.run([sys.executable, "-c", code, "--dataset", str(dataset_root), "--dataset-revision", "test",
                             "--episodes", "0", "1", "--baseline", "reference", "--action-hz", "30",
                             "--translation-frame", "body", "--pose-timing", "previous", "--output", str(output)],
                            cwd=Path(replay.__file__).parent, capture_output=True, text=True, timeout=30)
    assert result.returncode == 23, result.stderr
    assert json.loads((output / "episodes/episode-0-repeat-0.json").read_text())["status"] == "complete"
    assert json.loads((output / "metadata.json").read_text())["status"] == "running"
    assert not (output / "report.json").exists()
    assert not (output / "report.html").exists()


def test_failed_checkpoint_replacement_preserves_previous_result(tmp_path, monkeypatch):
    entry = {"episode_index": 0, "repeat": 0, "status": "complete", "windows": ["saved data"]}
    write_episode(tmp_path, entry)
    def fail_replace(*args):
        raise OSError("replacement failed")
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        write_episode(tmp_path, {**entry, "status": "failed"})
    assert json.loads((tmp_path / "episodes/episode-0-repeat-0.json").read_text()) == entry


def test_video_seek_honors_packed_episode_offset(tmp_path):
    path = tmp_path / "packed.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        for i in range(12):
            array = np.full((480,640,3), 20 if i < 6 else 220, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    assert video_frame(path, 0, 30).mean() < 40
    assert video_frame(path, 8/30, 30).mean() > 200
    with pytest.raises(ValueError, match="no video frame"):
        video_frame(path, 1, 30)


def test_nonintegral_rates_are_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        Settings(20, "body", "next").stride(30)


def test_prepare_fetches_only_selected_shards(dataset_root):
    from scripts.prepare_replay import prepare

    files = [str(p.relative_to(dataset_root)) for p in dataset_root.rglob("*.parquet")]
    files += [f"videos/observation.image.{h}/chunk-000/file-000.mp4" for h in ("left","right","center")]
    files += ["data/chunk-000/file-001.parquet"]
    fetched = []
    def fetch(name):
        fetched.append(name)
        return dataset_root / name
    shards = prepare([1], files, fetch)
    assert len(shards) == 3
    assert "data/chunk-000/file-001.parquet" not in fetched
    assert not any("center" in name for name in fetched)
    assert len(fetched) == 6


def test_failed_demonstration_is_rejected(dataset_root):
    dataset = Dataset(dataset_root)
    path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["task_success"] = False
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="task_success"):
        dataset.episode(0)


@pytest.mark.parametrize("ancestor", [".cache", ".git", "__pycache__"])
def test_submission_fingerprint_ignores_only_internal_cache_directories(tmp_path, ancestor):
    submission = tmp_path / ancestor / "snapshot"
    submission.mkdir(parents=True)
    policy = submission / "policy.py"
    policy.write_text("first policy")
    initial = replay.fingerprint(submission)
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "policy.py").write_text(policy.read_text())
    assert replay.fingerprint(copy) == initial
    for ignored in (".cache", ".git", "__pycache__"):
        (submission / ignored).mkdir()
        (submission / ignored / "ignored").write_text("cache contents")
    assert replay.fingerprint(submission) == initial
    policy.write_text("second policy")
    assert replay.fingerprint(submission) != initial


@pytest.mark.parametrize("interrupt", [False, True])
def test_cleanup_releases_container_even_when_log_capture_fails(monkeypatch, tmp_path, interrupt):
    def broken_logs(*args):
        raise KeyboardInterrupt() if interrupt else OSError("log directory unavailable")
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(replay.check, "_save_logs", broken_logs)
    monkeypatch.setattr(replay.subprocess, "run", run)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            replay.cleanup_container("owned-test-container", tmp_path / "log")
    else:
        warnings, removed = replay.cleanup_container("owned-test-container", tmp_path / "log")
        assert removed and "log directory unavailable" in warnings[0]
    assert commands == [["docker", "rm", "-f", "owned-test-container"]]


@pytest.mark.parametrize("inspection, safe", [("absent", True), ("present", False), ("failed", False), ("timeout", False)])
def test_failed_container_removal_requires_confirmed_absence(monkeypatch, tmp_path, inspection, safe):
    monkeypatch.setattr(replay.check, "_save_logs", lambda *args: None)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "rm":
            return subprocess.CompletedProcess(command, 1, "", "Docker removal failed")
        if inspection == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, int(inspection == "failed"),
            "owned-test-container\n" if inspection == "present" else "owned-test-container-other\n", "")
    monkeypatch.setattr(replay.subprocess, "run", run)
    warnings, removed = replay.cleanup_container("owned-test-container", tmp_path / "log")
    assert removed == safe
    assert "Docker removal failed" in warnings[0]
    assert len(warnings) == (1 if safe else 2)
    assert [command[1] for command in commands] == ["rm", "ps"]


@pytest.mark.parametrize("existing", ["directory", "file"])
def test_existing_output_is_a_cli_error(dataset_root, tmp_path, capsys, existing):
    output = tmp_path / "existing"
    if existing == "directory":
        output.mkdir()
        sentinel = output / "keep.txt"
    else:
        sentinel = output
    sentinel.write_text("existing result")
    with pytest.raises(SystemExit) as error:
        replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--episodes", "0",
                     "--baseline", "reference", "--action-hz", "30", "--translation-frame", "body",
                     "--pose-timing", "previous", "--output", str(output)])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "choose a new --output directory" in stderr and "Traceback" not in stderr
    assert sentinel.read_text() == "existing result"


def test_prepared_revision_mismatch_suppresses_score(dataset_root, tmp_path):
    (dataset_root / "replay_manifest.json").write_text(json.dumps({"status":"complete", "revision":"different"}))
    output = tmp_path / "mismatch"
    assert replay.main(["--dataset",str(dataset_root),"--dataset-revision","test","--episodes","0",
                        "--baseline","reference","--action-hz","30","--translation-frame","body",
                        "--pose-timing","previous","--output",str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert "revision differs" in report["error"]
    assert report["aggregate"] is None


@pytest.fixture
def suite_path(dataset_root, tmp_path):
    path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1].update(uuid="group-0", episode_id="group-0:1")
    pq.write_table(pa.Table.from_pylist(rows), path)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"version": 1, "dataset_repo": "test", "dataset_revision": "test",
        "total_open_tasks": 2, "uncovered_tasks": [{"id": 2, "name": "pens"}],
        "tasks": [{"id": 1, "name": "cup", "dataset_task": "cups", "steps": ["move the cup", "move the cup"],
                   "recordings": [{"uuid": "group-0", "episodes": [0, 1]}]}]}))
    return path


def test_suite_cli_preserves_repeated_step_positions_and_chunk_metrics(dataset_root, suite_path, tmp_path):
    output = tmp_path / "suite-report"
    assert replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--suite", str(suite_path),
                        "--baseline", "reference", "--action-hz", "30", "--translation-frame", "body",
                        "--pose-timing", "previous", "--repeats", "2", "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["aggregate"]["score"] == 100
    assert len(report["aggregate"]["tasks"][0]["steps"]) == 2
    assert report["coverage"]["selected_steps"] == 2 and report["coverage"]["total_open_tasks"] == 2
    assert len(report["episodes"]) == 4
    assert report["episodes"][0]["windows"][0]["summary"]["timesteps"] == 16
    assert report["suite_sha256"] == Suite(suite_path).sha256


@pytest.mark.parametrize("signal", ["observation", "action"])
@pytest.mark.parametrize("negative", [False, True])
def test_practice_gripper_filter_checks_raw_references(dataset_root, suite_path, tmp_path, signal, negative):
    spec = json.loads(suite_path.read_text())
    spec["gripper_filter"] = {"minimum_rad": -.1, "scope": "every episode in each recording"}
    suite_path.write_text(json.dumps(spec))
    if negative:
        path = dataset_root / "data/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[-1][signal + ".joint_states"][1] = -.11
        pq.write_table(pa.Table.from_pylist(rows), path)
    output = tmp_path / "filtered-report"
    code = replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--suite", str(suite_path),
                       "--baseline", "reference", "--action-hz", "30", "--translation-frame", "body",
                       "--pose-timing", "previous", "--output", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert code == (1 if negative else 0)
    if negative:
        assert report["aggregate"] is None
        assert not any(e["status"] == "complete" for e in report["episodes"])
        assert report["gripper_check"]["status"] != "passed"
    else:
        assert report["gripper_check"]["status"] == "passed"
        assert report["aggregate"]["normalized_rmse"] < 1e-10
        assert "Raw gripper values checked in every selected episode." in (output / "report.html").read_text()


@pytest.mark.parametrize("defect", ["missing_step", "duplicate_episode", "duplicate_task", "duplicate_uuid"])
def test_suite_rejects_incomplete_or_duplicated_membership(suite_path, defect):
    spec = json.loads(suite_path.read_text())
    task = spec["tasks"][0]
    if defect == "missing_step": task["recordings"][0]["episodes"].pop()
    if defect == "duplicate_episode": task["recordings"][0]["episodes"][1] = 0
    if defect == "duplicate_task": spec["uncovered_tasks"][0]["id"] = 1
    if defect == "duplicate_uuid": task["recordings"].append({"uuid": "group-0", "episodes": [2, 3]})
    suite_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError): Suite(suite_path)


@pytest.mark.parametrize("rule", [{"minimum_rad": -.1, "external_dependency": "missing"}, {}, {"minimum_rad": .1}])
def test_suite_rejects_unsupported_gripper_filters(suite_path, rule):
    spec = json.loads(suite_path.read_text())
    spec["gripper_filter"] = rule
    suite_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="gripper"):
        Suite(suite_path).validate_gripper_filter()


@pytest.mark.parametrize("defect", ["uuid", "step", "instruction", "outcome"])
def test_suite_validates_recording_step_and_frame_instructions(dataset_root, suite_path, defect):
    suite = Suite(suite_path)
    episode = Dataset(dataset_root).episode(1)
    assert suite.validate(episode)["step"] == 2
    if defect == "uuid": episode.metadata["uuid"] = "wrong"
    if defect == "step": episode.metadata["episode_id"] = "group-0:0"
    if defect == "instruction": episode.dataset.tasks[0] = "different action"
    if defect == "outcome": episode.metadata["success_short_horizon_task"] = False
    with pytest.raises(ValueError): suite.validate(episode)
    with pytest.raises(ValueError, match="revision"): suite.check_revision("different")


def test_suite_weights_repeats_recordings_steps_and_tasks_equally(suite_path):
    spec = json.loads(suite_path.read_text())
    spec["tasks"][0]["recordings"].append({"uuid": "group-1", "episodes": [2, 3]})
    spec["tasks"].append({"id": 2, "name": "pens", "dataset_task": "pens", "steps": ["place pen"],
                          "recordings": [{"uuid": "group-2", "episodes": [4]}]})
    spec["uncovered_tasks"] = []
    suite_path.write_text(json.dumps(spec))
    suite = Suite(suite_path)
    entries = []
    for index, scores in enumerate([[100, 0], [0, 0], [100, 100], [0, 0], [100, 100]]):
        for repeat, score in enumerate(scores):
            summary = metric_summary(score)
            entries.append({"episode_index": index, "repeat": repeat, "status": "complete", "summary": summary})
    result = aggregate_suite(entries, suite, 2)
    assert result["score"] == 68.75
    assert result["normalized_mse"] == 3125
    assert result["normalized_rmse"] == pytest.approx(np.sqrt(3125))
    assert result["position_cm"]["rmse"] == pytest.approx([np.sqrt(3125)] * 2)
    assert result["tasks"][0]["steps"][0]["summary"]["score"] == 75
    assert result["tasks"][0]["weakest_step"] == 2
    assert aggregate_suite(entries[:-1], suite, 2) is None
    assert aggregate_suite(entries + [entries[0]], suite, 2) is None
    entries[0]["status"] = "failed"
    assert aggregate_suite(entries, suite, 2) is None


def test_moving_score_aggregates_and_skips_levels_without_moving_timesteps(suite_path):
    suite = Suite(suite_path)
    entries = [{"episode_index": index, "repeat": 0, "status": "complete",
                "summary": {**metric_summary(50), "moving_score": moving, "stationary_score": 20}}
               for index, moving in enumerate([100, None])]
    result = aggregate_suite(entries, suite, 1)
    assert result["moving_score"] == 100 and result["stationary_score"] == 20
    assert result["tasks"][0]["steps"][1]["summary"]["moving_score"] is None
    for entry in entries:
        entry["summary"]["moving_score"] = None
    assert aggregate_suite(entries, suite, 1)["moving_score"] is None


def test_short_final_chunk_does_not_receive_a_full_chunks_weight(dataset_root):
    episode = Dataset(dataset_root).episode(0)
    settings = Settings(30, "body", "next")
    _, anchors, reference, grippers = next(episode.windows(settings))
    actions = reference_actions(anchors, reference, grippers, settings)
    good = compare(actions, anchors, reference, grippers, settings)
    actions[:, 14:] += 1
    bad = compare(actions, anchors, reference[:1], grippers[:1], settings)
    for w in (good, bad): w["latency_ms"] = 1
    summary = summarize([good, bad], settings)
    assert summary["score"] == pytest.approx(100 * 16 / 17)
    assert summary["gripper_rad"]["rmse"] == pytest.approx([np.sqrt(1 / 17)] * 2)
    assert summary["normalized_rmse"] == pytest.approx(np.sqrt(400 / (3 * 17)))


@pytest.mark.parametrize("cleanup_mode", ["ok", "warning_removed", "unsafe_first", "unsafe_last"])
def test_suite_resets_server_per_recording_and_preserves_completed_results(dataset_root, suite_path, tmp_path, monkeypatch, cleanup_mode):
    from umi_arena.replay_data import Episode

    meta_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(meta_path).to_pylist()
    rows += [{**r, "episode_index": r["episode_index"] + 2, "uuid": "group-2",
              "episode_id": f"group-2:{r['episode_index']}"} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows), meta_path)
    data_path = dataset_root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(data_path).to_pylist()
    rows += [{**r, "episode_index": r["episode_index"] + 2} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows), data_path)
    spec = json.loads(suite_path.read_text())
    spec["tasks"][0]["recordings"].append({"uuid": "group-2", "episodes": [2, 3]})
    suite_path.write_text(json.dumps(spec))
    submission = tmp_path / "submission"
    submission.mkdir(); (submission / "policy.py").write_text("# test adapter\n")
    starts, closes, cleanup = [], [], []
    def start(args, submission, log, report):
        starts.append(log.name); report.info["container"] = f"test-{len(starts)}"
    class Client:
        def __init__(self, *a, **k): pass
        def metadata(self, *a): return {"action_dim": 16}
        def infer(self, observation, **k):
            actions = np.zeros((16,16)); actions[:,[6,13]] = 1
            return {"actions": actions}, 1
        def close(self):
            closes.append(True)
    monkeypatch.setattr(Episode, "observation", lambda self, index: {
        f"observation.image.{hand}": np.zeros((480,640,3),np.uint8) for hand in ("left", "right")})
    monkeypatch.setattr(replay.check, "preflight", lambda *a: None)
    monkeypatch.setattr(replay.check, "image_digest", lambda *a: "test-image")
    monkeypatch.setattr(replay.check, "start_container", start)
    monkeypatch.setattr(replay.check, "Client", Client)
    def clean_container(name, log):
        cleanup.append(name)
        unsafe = cleanup_mode == "unsafe_first" or (cleanup_mode == "unsafe_last" and len(cleanup) == 4)
        return ([] if cleanup_mode == "ok" else ["cleanup warning"]), not unsafe
    monkeypatch.setattr(replay, "cleanup_container", clean_container)
    output = tmp_path / "server-suite"
    code = replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--suite", str(suite_path),
                        "--submission", str(submission), "--image", "test", "--checkpoint-id", "test",
                        "--action-hz", "30", "--translation-frame", "body", "--pose-timing", "previous",
                        "--repeats", "2", "--output", str(output)])
    report = json.loads((output / "report.json").read_text())
    stopped = cleanup_mode == "unsafe_first"
    assert code == (1 if stopped else 0)
    assert len(starts) == len(closes) == len(cleanup) == (1 if stopped else 4)
    assert len(report["episodes"]) == (2 if stopped else 8)
    assert (report["aggregate"] is None) == stopped
    assert all(e["status"] == "complete" for e in report["episodes"])
    if stopped:
        assert "stopping before the next recording" in report["error"]
    if cleanup_mode != "ok":
        assert report["warnings"] == ["cleanup warning"] * len(cleanup)
        assert "cleanup warning" in (output / "report.html").read_text()
    for entry in report["episodes"]:
        path = output / f"episodes/episode-{entry['episode_index']}-repeat-{entry['repeat']}.json"
        assert json.loads(path.read_text()) == entry


@pytest.mark.parametrize("slow", [False, True])
def test_cli_uses_runner_protocol_and_enforces_response_deadline(dataset_root, tmp_path, monkeypatch, slow):
    pytest.importorskip("openpi_client")
    from test_protocol import _start_runner
    from umi_arena.replay_data import Episode

    monkeypatch.setattr(Episode, "observation", lambda self, index: {
        INTER_HAND: self.data[INTER_HAND][index].astype(np.float32),
        "observation.joint_states": self.joints[index].astype(np.float32), "prompt": "move the cup",
        **{f"observation.image.{h}": np.zeros((480,640,3),np.uint8) for h in ("left","right")}})
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter.joinpath("policy.py").write_text(
        "import numpy as np, time\nclass Policy:\n"
        " def __init__(self, checkpoint_dir): pass\n"
        " def infer(self, obs):\n"
        "  assert len(obs) == 5 and obs['prompt'] == 'move the cup'\n"
        "  assert obs['observation.image.left'].shape == (480,640,3)\n"
        f"  time.sleep({0.3 if slow else 0})\n"
        "  a = np.zeros((32,16),np.float32); a[:,[6,13]]=1; a[:,14:]=.7; return a\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port = sock.getsockname()[1]
    process = _start_runner(adapter, port)
    try:
        output = tmp_path / "protocol-report"
        started = time.monotonic()
        code = replay.main(["--dataset",str(dataset_root),"--dataset-revision","test","--episodes","0",
                            "--server",f"ws://127.0.0.1:{port}","--checkpoint-id","test",
                            "--action-hz","30","--translation-frame","body","--pose-timing","next",
                            "--request-timeout", "0.05" if slow else "3", "--output",str(output)])
        report = json.loads((output / "report.json").read_text())
        assert code == (1 if slow else 0)
        if slow:
            assert "TimeoutError" in report["episodes"][0]["error"]
            assert report["aggregate"] is None
            assert time.monotonic() - started < 3
        else:
            assert report["episodes"][0]["windows"][0]["images"][0].startswith("data:image/jpeg")
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.parametrize("interrupted", [False, True])
def test_hashing_failure_preserves_original_error_and_raw_gripper_results(dataset_root, suite_path, tmp_path, monkeypatch, interrupted):
    spec = json.loads(suite_path.read_text())
    spec["gripper_filter"] = {"minimum_rad": -.1}
    suite_path.write_text(json.dumps(spec))
    def fail_hashes(self):
        raise OSError("dataset file unavailable")
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(Dataset, "hashes", fail_hashes)
    if interrupted:
        monkeypatch.setattr(replay, "evaluate_episode", interrupt)
    output = tmp_path / "hash-failure"
    assert replay.main(["--dataset", str(dataset_root), "--dataset-revision", "test", "--suite", str(suite_path),
                        "--baseline", "reference", "--action-hz", "30", "--translation-frame", "body",
                        "--pose-timing", "previous", "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["aggregate"] is None
    assert "dataset hashing failed: dataset file unavailable" in report["error"]
    assert ("evaluation interrupted" in report["error"]) == interrupted
    assert report["gripper_check"]["status"] == "passed"
    assert report["gripper_check"]["checked_episodes"] == report["gripper_check"]["expected_episodes"] == 2
