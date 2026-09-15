# UMI Arena evaluation runner

The fixed image that serves a submitted policy during the interface check and
the evaluation, the checker contestants run before submitting, and the two
reference adapters.

A submitted policy is **not** a websocket server. The runner is, and it calls the
submission through a small Python adapter. Teams publish weights to a public
Hugging Face repository plus a `policy.py`; they never touch websockets,
msgpack, or the metadata handshake.

```
[harness inference_node] ──ws / msgpack-numpy──> [runner]
                                                  ├ ws server + metadata handshake
                                                  ├ the submission, mounted read-only
                                                  └ call the team's Policy.infer(obs)
```

The observation keys, the action layout and the limits are specified on the
[Submission Format](../submission-format.html) page. This file covers the tooling.

## Layout

| Path | What it is |
|---|---|
| `check.py` | The checker. Starts the image on a submission and talks to it the way the harness does: handshake, action chunks, timing, prompt change, reconnect. Writes `report.json`. Contestants and organizers run the same file. |
| `replay.py` | Offline comparison of predicted trajectories with recorded LeRobot v3 episodes. Writes JSON and an interactive HTML report. Organizers score submissions with it. Teams can rehearse on the published tasks, described under [Practice evaluation](#practice-evaluation). |
| `suites/` | Fixed episode selections with task steps, source recording groups and dataset revisions. |
| `adapters/openpi_pi05/` | Reference adapter for `runtime: openpi`: a pi0.5 checkpoint served through the YUBI transforms, with the norm stats found under `assets/`. |
| `adapters/lerobot_hf/` | Reference adapter for `runtime: torch`: any LeRobot-family checkpoint (SmolVLA, GR00T N1.7, the pi0 ports, ACT, Diffusion). Resolves the policy class from `config.json`, and refuses a checkpoint whose cameras, state or action width do not match the contract. |
| `adapters/dummy/` | Returns a well-formed constant action chunk. No GPU, no model; for testing the contract itself. |
| `runner.py` | The runner. Websocket server, msgpack-numpy codec, metadata handshake, adapter loading, action validation, observation capture. |
| `entrypoint.sh` | Reads `runtime:` from the mounted submission's `umi_arena.yaml` (`openpi` when absent), exits 2 if it is not the runtime this image carries, and starts the runner in the image's venv. |
| `umi_arena/` | Helpers shipped inside the image so submissions do not vendor them: `umi_arena.yubi` (the YUBI transforms and data config, against upstream openpi) and `discover_asset_id`. |
| `docker/` | `Dockerfile.base` plus one `Dockerfile.<runtime>` per runtime, and `build.sh`, which builds them with one tag. Each runtime image runs its own import check at build time. |
| `scripts/` | Builds YUBI-shaped test fixtures from `lerobot/smolvla_base` and `nvidia/GR00T-N1.7-3B`: weights untouched, feature shapes and stats set to the contract. Pipeline fixtures, not policies. |
| `tests/` | The test suite, in layers that skip without their prerequisites. See `tests/README.md`. |
| `tests/broken/` | Adapters broken on purpose, one per defect the checker must name. |
| `verify_registry_bypass.py` | Shows that openpi loads a checkpoint from an in-process `TrainConfig`, with nothing written into the openpi tree. |

Each adapter directory carries a `umi_arena.yaml` declaring `runtime: openpi` or
`runtime: torch`, which selects the image it runs in; the entrypoint refuses a
mismatch.

## Prepare a submission folder

Run the commands below from `evaluation-runner/`. Set `submission` to your saved
checkpoint directory. It must contain `policy.py`, the model weights and any
configuration or normalization assets the model needs.

For a compatible OpenPI/pi0.5 checkpoint, add the reference adapter:

```sh
submission=/absolute/path/to/my-submission
cp adapters/openpi_pi05/policy.py adapters/openpi_pi05/umi_arena.yaml "$submission/"
```

```text
my-submission/
├── policy.py
├── umi_arena.yaml             # runtime: openpi
├── params/                    # Saved checkpoint parameters
└── assets/
    └── <asset-id>/
        └── norm_stats.json
```

For a compatible LeRobot checkpoint, use the Torch adapter instead:

```sh
submission=/absolute/path/to/my-submission
cp adapters/lerobot_hf/policy.py adapters/lerobot_hf/umi_arena.yaml "$submission/"
```

```text
my-submission/
├── policy.py
├── umi_arena.yaml             # runtime: torch
├── config.json
├── model.safetensors
└── ...                       # Saved processor configs, statistics and other model assets
```

- Keep the complete checkpoint structure; required files depend on the model format.
- Copy a reference adapter only when its model configuration matches your checkpoint.
  If you already have a custom `policy.py`, keep it and check the contract below.
- The runner mounts the folder read-only and passes its path to `Policy(checkpoint_dir)`.
  Keep datasets and generated reports outside this folder.
- Use the matching `umi-arena-openpi:<round-tag>` or `umi-arena-torch:<round-tag>` image.
  If it is not available locally, see [Building it](#building-it), then run [the checker](#the-checker).

## The adapter contract

```python
class Policy:
    def __init__(self, checkpoint_dir: str): ...
    def infer(self, obs: dict) -> np.ndarray:
        """obs: the harness's observation dict, verbatim.
        returns: (chunk, 16); the runner casts to float32."""
```

Two things that trip people up:

- **Images arrive at native camera resolution, `uint8 (480, 640, 3)`, not
  224×224.** Resizing is the adapter's job; openpi's `ResizeImages` and LeRobot's
  own preprocessing both already do it.
- **The first inference on a freshly loaded model is slow**, JIT plus the first
  cuDNN autotune, and can outlast the websocket's 20 s keepalive. The adapter
  must pre-warm in `__init__`; both reference adapters show how.

## The checker

```sh
uv run check.py --submission ./my-submission --tag <round-tag>            # a local directory
uv run check.py --hf team/model --revision <sha> --tag <round-tag>        # the pinned revision, as at intake
```

Needs Linux, Docker with the NVIDIA container toolkit, a GPU, and `uv`. The
verdict and every check go to `report.json`, the container's log next to it; exit
0 when every check passed, 1 when one failed. The checker never imports the
submission's code and does not judge the actions.

The real robot sends only the left and right wrist images. A submission must
run without `observation.image.center`, so the checker omits it.

- `--tag` names the round's image tag; the default `dev` exists only on a build machine.
- `--env HF_TOKEN=<token>` forwards a Hugging Face token for a gated backbone.
  Nothing is forwarded into the container without a flag.
- `--hf-cache <dir>` mounts the cache's `hub/` and `modules/` read-only with
  `HF_HUB_OFFLINE=1`, so the container needs neither token nor network; the login
  files beside them stay out. Warm the cache once on the host:
  `HF_HOME=<dir> uvx --from huggingface_hub huggingface-cli download <backbone>`.
- `--fixtures <file>` replays recorded observations, as `runner.py --fixture-out`
  writes them, instead of synthetic ones. Keys outside the observation contract
  are removed before inference.

## Practice evaluation

`replay.py` sends recorded observations to your policy and compares the action chunks it
returns with what the demonstrator did. It needs the released dataset and the same Linux,
Docker and GPU host the checker needs.

### Run the practice suite

Prepare your submission and pass the checker first. The example below uses OpenPI;
use `umi-arena-torch` for a Torch submission. Replace `MODEL_REVISION` with your
checkpoint identifier and `ROUND_TAG` with the image tag you have installed.
The timing and translation settings must match your checkpoint.

```sh
revision=27fc30ae498d43aa6610918c6c77d7cec8267781
uv run scripts/prepare_replay.py --revision "$revision" \
  --suite suites/practice-cup-smartphone.json --output replay-data/practice

uv run replay.py --dataset replay-data/practice --dataset-revision "$revision" \
  --suite suites/practice-cup-smartphone.json --baseline hold --action-hz 30 \
  --translation-frame body --pose-timing previous --output replay-results/hold

uv run replay.py --dataset replay-data/practice --dataset-revision "$revision" \
  --suite suites/practice-cup-smartphone.json --submission "$submission" \
  --checkpoint-id MODEL_REVISION --image umi-arena-openpi:ROUND_TAG --action-hz 30 \
  --translation-frame body --pose-timing previous --output replay-results/model
```

`suites/practice-cup-smartphone.json` fixes five recordings for each of the two open tasks
whose recordings are published: 50 episodes in total. Pass `--suite` explicitly to select it.
The downloader prints the dataset SHA it resolved and saves it in
`replay-data/practice/replay_manifest.json`.

For model evaluation, the full practice suite requires `--submission` and `--image`.
Replay starts a fresh policy container for each recording and repeat, sharing that
container across the recording's task steps. It stops the container afterward.

### Use a dataset already on disk

Skip `prepare_replay.py` and point `--dataset` at the local LeRobot v3.0 root.
It must contain `meta/info.json`, task and episode metadata, episode data, and the
left/right camera videos for every selected episode. Keep their original directory layout.

For the practice suite, the data must be from its pinned revision:

```sh
dataset=/absolute/path/to/local/dataset
revision=27fc30ae498d43aa6610918c6c77d7cec8267781

uv run replay.py --dataset "$dataset" --dataset-revision "$revision" \
  --suite suites/practice-cup-smartphone.json --submission "$submission" \
  --checkpoint-id MODEL_REVISION --image umi-arena-openpi:ROUND_TAG --action-hz 30 \
  --translation-frame body --pose-timing previous --output replay-results/local-suite
```

Use the submission folder and matching image from the setup above. If the dataset has a
`replay_manifest.json`, its revision must also match `--dataset-revision`.

### Connect to an existing policy server

Use `--server` when your policy server is already running, including in your own container.
It must implement the runner's WebSocket protocol and accept the observation/action contract.
Replay connects to it without starting or stopping a container.

Start the server with fresh policy state before each invocation. `--server` accepts
exactly one episode and one repeat, so it cannot run the full practice suite.
Use managed containers as above for the full suite, so policy state is reset between recordings.

This example selects episode `61164` from the practice dataset already on disk:

```sh
dataset=/absolute/path/to/local/dataset
revision=27fc30ae498d43aa6610918c6c77d7cec8267781

uv run replay.py --dataset "$dataset" --dataset-revision "$revision" \
  --episodes 61164 --server ws://127.0.0.1:8124 \
  --checkpoint-id MODEL_REVISION --action-hz 30 \
  --translation-frame body --pose-timing previous --output replay-results/server-61164
```

Replace the address with your server's reachable host and port. For a different local
dataset, use its revision identifier and an episode ID present in its metadata.

### Open the report

Open `report.html` in an output directory to inspect a chunk, a recording or the worst
error. If evaluation ran remotely, copy that file to your computer. It contains its data,
images, CSS and JavaScript, so you can open it directly in a browser without a server.
Choose a new `--output` directory for each run; existing output paths are rejected.

Completed episodes are saved individually in `episodes/` during the run, with run
settings in `metadata.json`. The complete `report.json` and self-contained `report.html`
are generated once when evaluation finishes, fails, or is interrupted with Ctrl+C.
If the process is killed, completed episode files remain available; automatic resume
is not supported.

Cleanup warnings do not discard completed results. Evaluation stops before the next
recording if it cannot confirm that the previous policy container was removed.

- Use `--episodes <id> ...` instead of `--suite` to inspect individual episodes.
  The downloader accepts the same selection.
- `--repeats <count>` reruns the selection with a fresh process per recording and repeat
  when using `--submission`. Every repeat contributes to the result.
- The scoring scales and tolerances are fixed so runs use the same error weights.
  Action frequency, translation frame and pose timing must match the checkpoint.

### What it scores

- Every returned chunk starts from the recorded hand poses, so a prediction is compared
  with the demonstration at each of its timesteps.
- Six errors per timestep: position, orientation and gripper for each hand. Position is the
  distance between the predicted and the recorded hand root in centimetres, orientation is
  the shortest rotation angle in degrees, and gripper is the joint-angle error in radians.
- The headline number is a normalized RMSE. Each error is divided by 2 cm, 5 degrees and
  0.05 rad, and the six squared ratios are averaged with equal weight. Lower is better, and
  0 means the prediction matched the demonstration.
- Run `--baseline hold` first. It predicts no motion at all, so it is the number a policy has
  to beat, and a score on its own says little.

### What it does not tell you

- The published recordings are training data, so a practice score is optimistic.
- It measures agreement with one demonstration. It is not task success, and a different but
  valid trajectory counts as error.
- Error accumulates inside a chunk. Large error in the first rows points at a convention or
  normalization problem, while error that only grows with look-ahead is ordinary prediction
  difficulty.
- The practice suite covers two of the five open tasks. The other three are listed as
  uncovered and receive no score.

## The images

```
umi-arena-base:<round-tag>    OS packages + uv

umi-arena-openpi:<round-tag>   FROM base   /opt/venv/openpi   Python 3.11, JAX 0.5.3 + upstream openpi
umi-arena-torch:<round-tag>    FROM base   /opt/venv/torch    Python 3.12, torch (cu128) + lerobot 0.6 [groot,smolvla,pi]
```

Each runtime image contains `/opt/umi_arena/` with the runner and helpers,
and `/entrypoint.sh` to select and validate the submission's runtime.

```sh
docker run --gpus all --network host \
  -v /path/to/submission:/submission:ro \
  umi-arena-<runtime>:<round-tag>
```

A submission that declares the other runtime exits 2 with a message naming
both. One that declares none runs on `umi-arena-openpi`.
An empty or malformed runtime declaration exits 2 instead of selecting a default.

The two runtimes are separate environments. openpi pins a LeRobot revision with
no model implementations, so a LeRobot checkpoint cannot be loaded from the
openpi venv; the torch venv carries a current LeRobot instead. Both images get
`openpi-client`, because the msgpack-numpy codec the websocket protocol is
defined in belongs to the protocol, not to openpi. Which policy families a torch
submission may use is decided by the LeRobot extras installed there; adding a
family means adding its extra and re-tagging the image.

### Building it

```sh
cd evaluation-runner
docker/build.sh <round-tag>            # base, then openpi and torch
docker/build.sh <round-tag> torch      # one runtime only
```

Set `OPENPI_REF=<sha>` in the environment to build against another upstream
openpi commit.
The build script supplies the same commit and base tag to both runtime images.
Direct Docker builds must supply both `BASE` and `OPENPI_REF` as build arguments.

The build works around three things:

| Symptom | Cause | Workaround in the Dockerfiles |
|---|---|---|
| `external filter 'git-lfs filter-process' failed` | uv clones openpi's pinned `lerobot` git rev and git-lfs tries to smudge it | `GIT_LFS_SKIP_SMUDGE=1` |
| `No such file or directory: 'clang'` | some openpi dependencies have no wheel and compile from source | install `build-essential clang cmake` |
| `ModuleNotFoundError: transformers` | LeRobot's modeling modules import it; it comes with the policy extras | install `lerobot[groot,smolvla,pi]` |

The openpi image runs an import check at build time that fails the build if the
`jax==0.5.3` pin slips; a newer jax breaks orbax checkpoint loading. The
environment is resolved at build time; a lock file published with each round's
image tag is planned.

## Recorded observations

`runner.py --fixture-out` records the observation dicts it receives, for the
checker's `--fixtures`. They are not committed here: the images are real scene
captures covered by the data-sharing agreement.
