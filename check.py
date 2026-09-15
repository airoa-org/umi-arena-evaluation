#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "websockets>=13",
#   "msgpack",
#   "huggingface_hub",
#   "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client",
# ]
# ///
"""UMI Arena submission checker: can this submission run under the contract?

Starts the runtime image on a submission, then talks to it the way
the harness's inference_node does: websocket, msgpack-numpy, metadata first,
one observation per request. It never imports the team's code, and it does not
judge the actions. A contestant runs it before submitting; the organizers run
the same file at intake.

    uv run check.py --submission ./my-submission
    uv run check.py --hf team/model --revision <sha>       # organizers: fetch, then the same run

Exit 0 when every check passes (warnings allowed), 1 on a failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)  # websockets' sync API nags about context managers

CHECKER_VERSION = "1"

# --- the contract, as inference_node sends and reads it; tests/test_check.py keeps it in step with runner.py ---
ACTION_DIM = 16
MIN_CHUNK = 16
IMAGE_SHAPE = (480, 640, 3)
OBSERVATION = {
    "observation.image.left": ("uint8", IMAGE_SHAPE),
    "observation.image.right": ("uint8", IMAGE_SHAPE),
    "observation.pose.left_hand_root_to_right_hand_root.absolute": ("float32", (7,)),
    "observation.joint_states": ("float32", (2,)),
    "prompt": ("str", ()),
}
PROMPTS = [
    "Pick up the cup with your right hand and set it on the plate",
    "Remove the top shorter curved LEGO brick",
]
DEFAULT_RUNTIME = "openpi"
HANDSHAKE_TIMEOUT_S = 10.0
SLOW_MS = 1600          # the per-call figure submission-format.html publishes; advisory
FIRST_CALL_RATIO = 5.0  # first call this much slower than the median means no pre-warm


def synthetic_observation(seed: int = 0, prompt: str = PROMPTS[0]) -> dict:
    """Only the real robot's contract keys, so simulator-only dependencies fail here."""
    rng = np.random.default_rng(seed)
    obs = {}
    for key, (dtype, shape) in OBSERVATION.items():
        if dtype == "str":
            obs[key] = prompt
        elif dtype == "uint8":
            obs[key] = rng.integers(0, 256, size=shape, dtype=np.uint8)
        elif key.startswith("observation.pose."):
            # A plausible inter-hand pose, so a normalising adapter sees in-distribution state.
            obs[key] = np.array([0.23, -0.05, 0.27, -0.58, -0.09, 0.72, 0.34], dtype=np.float32)
        else:
            obs[key] = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    return obs


# --- report --------------------------------------------------------------------

class CheckFailed(Exception):
    """A check failed and the stage cannot continue."""


@dataclass
class Report:
    checks: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append({"name": name, "status": status, "detail": detail})
        mark = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}[status]
        print(f"{mark}  {name}" + (f"  — {detail}" if detail else ""), flush=True)

    def ok(self, name, detail=""):
        self.add(name, "pass", detail)

    def warn(self, name, detail=""):
        self.add(name, "warn", detail)

    def fail(self, name, detail=""):
        self.add(name, "fail", detail)
        raise CheckFailed(name)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "fail" for c in self.checks)

    def verdict(self) -> str:
        n = {s: sum(c["status"] == s for c in self.checks) for s in ("pass", "warn", "fail")}
        return f"{'FAIL' if n['fail'] else 'PASS'}: {n['pass']} passed, {n['warn']} warnings, {n['fail']} failed"

    def write(self, path: Path) -> None:
        path.write_text(json.dumps({"checker_version": CHECKER_VERSION, "verdict": self.verdict(),
                                    **self.info, "checks": self.checks}, indent=2) + "\n")


# --- the client, as inference_node behaves ------------------------------------

class Client:
    def __init__(self, url: str, *, keepalive_s: float, open_timeout_s: float = 5.0):
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self._msgpack = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        # inference_node's keepalive (websockets default 20 s): a call that outlasts it drops there too.
        self._conn = websockets.sync.client.connect(
            url, compression=None, max_size=None, open_timeout=open_timeout_s,
            ping_interval=keepalive_s, ping_timeout=keepalive_s)

    def metadata(self, timeout_s: float) -> dict:
        return self._msgpack.unpackb(self._conn.recv(timeout=timeout_s))

    def infer(self, obs: dict, *, timeout_s: float | None = None) -> tuple[dict, float]:
        """Returns (response, round_trip_ms). A str response is the server's error text."""
        import websockets.exceptions

        t0 = time.monotonic()
        try:
            self._conn.send(self._packer.pack(obs))
            raw = self._conn.recv() if timeout_s is None else self._conn.recv(timeout=timeout_s)
        except websockets.exceptions.ConnectionClosed as exc:
            raise ConnectionError(f"connection closed during the call after {time.monotonic() - t0:.1f}s "
                                  f"({exc.code if hasattr(exc, 'code') else exc})") from exc
        dt = (time.monotonic() - t0) * 1000.0
        if isinstance(raw, str):
            return {"error": raw}, dt
        return self._msgpack.unpackb(raw), dt

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# --- stages ----------------------------------------------------------------------

def parse_runtime(submission: Path) -> str | None:
    """The entrypoint's rule: first `runtime:` line wins; none means the default."""
    yaml = submission / "umi_arena.yaml"
    if not yaml.is_file():
        return None
    for line in yaml.read_text().splitlines():
        if re.match(r"^\s*runtime\s*:", line):
            m = re.fullmatch(r"""\s*runtime\s*:\s*([A-Za-z0-9_-]+|"[A-Za-z0-9_-]+"|'[A-Za-z0-9_-]+')(?:\s+#.*)?\s*""", line)
            if m is None:
                raise ValueError(f"invalid runtime declaration in {yaml}; expected runtime: openpi or runtime: torch")
            return m.group(1).strip("\"'")
    return None


def preflight(args, report: Report) -> None:
    if not shutil.which("docker"):
        report.fail("docker is installed", "docker not found on PATH")
    report.ok("docker is installed")
    if platform.system() != "Linux":
        report.fail("host can run the image", f"{platform.system()}: the runtime images need Linux with an NVIDIA GPU")
    r = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"      pulling {args.image} …", flush=True)
        r = subprocess.run(["docker", "pull", args.image], capture_output=True, text=True)
        if r.returncode != 0:
            report.fail("image is available", r.stderr.strip().splitlines()[-1] if r.stderr else "pull failed")
    report.ok("image is available", args.image)
    r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "nvidia-smi", args.image, "-L"],
                       capture_output=True, text=True, timeout=120)
    lines = (r.stdout if r.returncode == 0 and r.stdout.strip() else r.stderr or r.stdout).strip().splitlines()
    if r.returncode != 0 or not lines:
        report.fail("GPU is reachable from a container", lines[-1] if lines else f"nvidia-smi exit {r.returncode}, no output")
    report.ok("GPU is reachable from a container", lines[0])


def image_digest(image: str) -> str:
    r = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}} {{.RepoDigests}}"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def container_command(args, name: str, submission: Path) -> list[str]:
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--network", "host",
           "-e", f"UMI_ARENA_PORT={args.port}", "-v", f"{submission}:/submission:ro"]
    for e in args.env:
        cmd += ["-e", e]
    if args.hf_cache:  # only the model files: HF_HOME also holds the login token, which must not reach the submission's code
        for sub in ("hub", "modules"):
            if (args.hf_cache / sub).is_dir():
                cmd += ["-v", f"{(args.hf_cache / sub).resolve()}:/hf-cache/{sub}:ro"]
        cmd += ["-e", "HF_HOME=/hf-cache", "-e", "HF_HUB_OFFLINE=1"]
    for v in args.volume:
        cmd += ["-v", v]
    cmd.append(args.image)
    return cmd


def start_container(args, submission: Path, log_path: Path, report: Report) -> str:
    name = f"umi-check-{args.port}"
    leftover = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}} {{.Status}}"],
                              capture_output=True, text=True).stdout.strip()
    if leftover:
        report.fail("no earlier checker container is in the way", f"{leftover}; remove it with: docker rm -f {name}")
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            report.fail("port is free", f"something already listens on 127.0.0.1:{args.port}; pass --port")
    cmd = container_command(args, name, submission)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        report.fail("container starts", r.stderr.strip())
    report.info["container"] = name  # set before any wait, so a failed load is still cleaned up
    t0 = time.monotonic()
    deadline = t0 + args.load_timeout
    import websockets.sync.client
    while time.monotonic() < deadline:
        state = subprocess.run(["docker", "inspect", name, "--format", "{{.State.Status}} {{.State.ExitCode}}"],
                               capture_output=True, text=True).stdout.split()
        if state and state[0] != "running":
            _save_logs(name, log_path)
            text = log_path.read_text()
            tail = "\n".join(text.splitlines()[-40:])
            hint = ""
            if "GatedRepoError" in text or "401 Client Error" in text:
                repo = re.search(r"huggingface\.co/([\w.-]+/[\w.-]+)/resolve", text)
                hint = (f"\nhint: the model needs a gated Hugging Face repo{' (' + repo.group(1) + ')' if repo else ''}. "
                        "For your own submission pass --env HF_TOKEN=<token>; at intake warm a cache on the host "
                        "and pass --hf-cache <dir>, so no token enters the container")
            report.fail("runner comes up", f"container exited with code {state[1]}; last log lines:\n{tail}{hint}")
        try:
            with websockets.sync.client.connect(f"ws://127.0.0.1:{args.port}", open_timeout=1):
                pass
            break
        except Exception:
            time.sleep(1.0)
    else:
        _save_logs(name, log_path)
        report.fail("runner comes up", f"no listener on port {args.port} after {args.load_timeout}s (load + pre-warm); see {log_path}")
    load_s = time.monotonic() - t0
    report.info["load_s"] = round(load_s, 1)
    report.ok("runner comes up", f"{load_s:.1f}s to listening, including pre-warm")
    return name


def _save_logs(name: str, log_path: Path) -> None:
    r = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    log_path.write_text(r.stdout + r.stderr)


def stop_container(name: str, log_path: Path, keep: bool) -> None:
    _save_logs(name, log_path)
    if not keep:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def check_actions(out: dict, report: Report, label: str) -> np.ndarray:
    if "error" in out:
        report.fail(f"{label}: server answered", "server error:\n" + out["error"].rstrip()[-1500:])
    if "actions" not in out:
        report.fail(f"{label}: response has actions", f"keys: {sorted(out)}")
    a = np.asarray(out["actions"])
    if a.ndim != 2:
        report.fail(f"{label}: actions are 2-D", f"shape {a.shape}")
    if a.shape[1] != ACTION_DIM:
        report.fail(f"{label}: actions have {ACTION_DIM} columns", f"shape {a.shape}")
    if a.shape[0] < MIN_CHUNK:
        report.fail(f"{label}: actions have at least {MIN_CHUNK} rows", f"shape {a.shape}")
    if not np.isfinite(a).all():
        report.fail(f"{label}: actions are finite", f"{int((~np.isfinite(a)).sum())} non-finite values")
    return a


def run_protocol(url: str, observations: list[dict], args, report: Report, connect=None) -> None:
    connect = connect or Client  # tests pass a fake with metadata(), infer() and close()
    # --- handshake ---------------------------------------------------------------
    try:
        client = connect(url, keepalive_s=args.keepalive)
    except Exception as exc:
        report.fail("client connects", str(exc))
    try:
        try:
            meta = client.metadata(args.handshake_timeout)
        except TimeoutError:
            report.fail("metadata arrives on connect", f"nothing within {args.handshake_timeout}s; inference_node blocks here at startup")
        except Exception as exc:
            report.fail("metadata arrives on connect", str(exc))
        if not isinstance(meta, dict):
            report.fail("metadata is a dict", repr(meta)[:200])
        if meta.get("action_dim") != ACTION_DIM:
            report.fail(f"metadata declares action_dim {ACTION_DIM}", repr(meta)[:200])
        report.ok("metadata arrives on connect", ", ".join(f"{k}={v}" for k, v in meta.items())[:160])
        report.info["metadata"] = meta

        # --- contract ------------------------------------------------------------
        rt_ms, infer_ms, shapes = [], [], set()
        for i, obs in enumerate(observations):
            try:
                out, dt = client.infer(obs)
            except ConnectionError as exc:
                report.fail(f"call {i}: connection stays open", str(exc))
            a = check_actions(out, report, f"call {i}")
            shapes.add(a.shape)
            rt_ms.append(dt)
            infer_ms.append(float(out.get("server_timing", {}).get("infer_ms", np.nan)))
            if a.dtype != np.float32:
                report.warn(f"call {i}: actions are float32", f"{a.dtype}; the runner casts, but float32 is the contract")
        n = len(observations)
        report.ok(f"{n} calls return valid action chunks", f"shapes {sorted(shapes)}")

        # --- timing ----------------------------------------------------------------
        rt = np.array(rt_ms)
        stats = {"first_ms": round(rt[0], 1), "p50_ms": round(float(np.percentile(rt, 50)), 1),
                 "p95_ms": round(float(np.percentile(rt, 95)), 1), "max_ms": round(float(rt.max()), 1),
                 "server_p50_ms": round(float(np.nanmedian(infer_ms)), 1) if not np.all(np.isnan(infer_ms)) else None}
        report.info["timing"] = stats
        median = stats["p50_ms"]
        if n > 1 and rt[0] > FIRST_CALL_RATIO * median and rt[0] > args.slow_ms:
            report.warn("first call is pre-warmed", f"first call {rt[0]:.0f} ms, median {median:.0f} ms: the adapter does not pre-warm in __init__; the harness's first episode would stall or drop")
        else:
            report.ok("first call is pre-warmed", f"first {rt[0]:.0f} ms, median {median:.0f} ms")
        if stats["p95_ms"] > args.slow_ms:
            report.warn("calls are within the published budget", f"p95 {stats['p95_ms']:.0f} ms > {args.slow_ms} ms; the robot halts between calls for that long")
        else:
            report.ok("calls are within the published budget", f"p50 {median:.0f} ms, p95 {stats['p95_ms']:.0f} ms, max {stats['max_ms']:.0f} ms")

        # --- robustness ------------------------------------------------------------
        other = dict(observations[0]); other["prompt"] = PROMPTS[1] if observations[0].get("prompt") != PROMPTS[1] else PROMPTS[0]
        try:
            out, _ = client.infer(other)
        except ConnectionError as exc:
            report.fail("prompt can change mid-connection", str(exc))
        check_actions(out, report, "prompt change")
        report.ok("prompt can change mid-connection")
    finally:
        client.close()

    try:
        client = connect(url, keepalive_s=args.keepalive)
        meta2 = client.metadata(args.handshake_timeout)
        out, _ = client.infer(observations[0])
        client.close()
    except Exception as exc:
        report.fail("reconnect works", f"{exc}; inference_node reconnects after a timeout and expects metadata again")
    if not isinstance(meta2, dict) or meta2.get("action_dim") != ACTION_DIM:
        report.fail("reconnect works", "metadata not re-sent on the second connection")
    check_actions(out, report, "after reconnect")
    report.ok("reconnect works")


def load_fixtures(path: Path) -> list[dict]:
    """Load {"observations": [obs, ...]} with only the real robot's contract keys."""
    from openpi_client import msgpack_numpy
    data = msgpack_numpy.unpackb(path.read_bytes())
    obs = data.get("observations") if isinstance(data, dict) else None
    if not obs:
        raise ValueError(f"{path}: expected a msgpack map with a non-empty 'observations' list")
    return [{key: value for key, value in dict(o).items() if key in OBSERVATION} for o in obs]


def fetch_hf(repo: str, revision: str | None) -> tuple[Path, str]:
    """Fetch into $UMI_ARENA_CACHE/submissions/<repo>@<sha> (default ~/umi-arena-cache),
    reused on the next run. Not /tmp and not a dot-directory: a snap-confined Docker
    cannot bind-mount either."""
    from huggingface_hub import snapshot_download, HfApi
    sha = HfApi().model_info(repo, revision=revision).sha
    cache = Path(os.environ.get("UMI_ARENA_CACHE", Path.home() / "umi-arena-cache"))
    dest = cache / "submissions" / f"{repo.replace('/', '__')}@{sha}"
    local = snapshot_download(repo, revision=sha, local_dir=dest)
    return Path(local), sha


# --- main -------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The submission's policy.py runs inside the container with the host network, any --volume and "
               "any --env you pass; it is not sandboxed. Give it nothing the team must not have: at intake "
               "use --hf-cache instead of a token.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--submission", type=Path, help="local submission directory")
    src.add_argument("--hf", help="Hugging Face model repo to fetch, e.g. team/model")
    p.add_argument("--revision", help="commit to fetch with --hf (default: main)")
    p.add_argument("--image", help="runtime image; default umi-arena-<runtime>:<tag>")
    p.add_argument("--tag", default="dev", help="image tag when --image is not given")
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--calls", type=int, default=20, help="observations to send")
    p.add_argument("--fixtures", type=Path, help="msgpack file {'observations': [...]} as runner.py --fixture-out writes, instead of synthetic ones")
    p.add_argument("--load-timeout", type=float, default=600.0, help="seconds to wait for the runner, load + pre-warm")
    p.add_argument("--handshake-timeout", type=float, default=HANDSHAKE_TIMEOUT_S)
    p.add_argument("--keepalive", type=float, default=20.0, help="websocket ping interval/timeout, as inference_node uses")
    p.add_argument("--slow-ms", type=float, default=SLOW_MS, help="warn when p95 exceeds this")
    p.add_argument("--env", action="append", default=[], help="extra -e for the container, e.g. HF_TOKEN=<token> for your own gated model")
    p.add_argument("--volume", action="append", default=[], help="extra -v for the container, e.g. ~/hf:/hfhome")
    p.add_argument("--hf-cache", type=Path, help="pre-warmed HF_HOME: its hub/ and modules/ are mounted read-only with HF_HUB_OFFLINE=1, its login files are not (intake)")
    p.add_argument("--keep", action="store_true", help="leave the container running after the run")
    p.add_argument("--report", type=Path, default=Path("report.json"))
    args = p.parse_args(argv)
    if args.calls < 1:
        p.error("--calls must be at least 1")
    if args.hf_cache and not (args.hf_cache / "hub").is_dir():
        p.error(f"--hf-cache {args.hf_cache} has no hub/ directory; warm it first (HF_HOME=<dir> huggingface-cli download <backbone>)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    report = Report()
    report.info["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log_path = args.report.with_suffix(".log")  # the container log, next to the report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.fixtures:
            try:
                observations = load_fixtures(args.fixtures)[: args.calls]
            except Exception as exc:
                report.fail("fixtures load", f"{args.fixtures}: {exc}")
            report.info["observations"] = f"fixtures {args.fixtures} ({len(observations)})"
        else:
            observations = [synthetic_observation(i, PROMPTS[0]) for i in range(args.calls)]
            report.info["observations"] = f"synthetic ({len(observations)})"

        if args.hf:
            try:
                submission, sha = fetch_hf(args.hf, args.revision)
            except Exception as exc:
                report.fail("submission fetched from the Hub", f"{args.hf}: {exc}")
            report.info["submission"] = f"{args.hf}@{sha}"
        else:
            submission = args.submission.resolve()
            report.info["submission"] = str(submission)
        if not submission.is_dir():
            report.fail("submission directory exists", str(submission))
        try:
            runtime = parse_runtime(submission)
        except ValueError as exc:
            report.fail("umi_arena.yaml declares a runtime", str(exc))
        report.info["runtime"] = runtime or f"{DEFAULT_RUNTIME} (default: no runtime line)"
        if runtime is None:
            report.warn("umi_arena.yaml declares a runtime", f"none found; the image defaults to {DEFAULT_RUNTIME}")
            runtime = DEFAULT_RUNTIME
        args.image = args.image or f"umi-arena-{runtime}:{args.tag}"
        preflight(args, report)
        report.info["image"] = args.image
        report.info["image_digest"] = image_digest(args.image)
        start_container(args, submission, log_path, report)
        run_protocol(f"ws://127.0.0.1:{args.port}", observations, args, report)
    except CheckFailed:
        pass
    except KeyboardInterrupt:
        report.add("interrupted", "fail", "")
    except Exception:  # the report is the checker's only output; a checker bug must land there too
        report.add("checker error", "fail", traceback.format_exc())
    finally:
        if report.info.get("container"):
            stop_container(report.info["container"], log_path, args.keep)
            report.info["runner_log"] = str(log_path)
    report.write(args.report)
    print(f"\n{report.verdict()}  ->  {args.report}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
