# Tests

Layers that skip cleanly when their prerequisites are absent, so the suite runs
on a laptop, on a build machine with the images, and on a GPU box.

| Layer | File | Needs | What it protects |
|---|---|---|---|
| unit | `test_unit_runner.py` | nothing | The contract the runner enforces on a submission's output, and how adapter-loading failures surface |
| unit | `test_unit_helpers.py` | nothing | `discover_asset_id`, and the entrypoint's `runtime:` parser |
| unit | `test_unit_build.py` | nothing | Both runtime builds receive the same base tag and openpi commit; the checker uses the same client revision |
| unit | `test_unit_yubi.py` | openpi | The YUBI transforms: two wrist cameras and a masked base slot, action padding and trimming; run inside the image by `test_image.py` |
| unit | `test_unit_check.py` | nothing | `check.py`'s host-side logic: the runtime parser, the docker command, contract-only observations and fixtures, and that every outcome reaches the report |
| unit | `test_unit_adapter.py` | nothing | The lerobot adapter's feature check, without torch |
| protocol | `test_protocol.py` | `websockets`, `openpi_client` | The websocket exchange against a live runner: handshake, chunking, reconnect, the error frame, fixture capture; run inside the image by `test_image.py` |
| image | `test_image.py` | built images | What the images contain, entrypoint behaviour on malformed submissions, and the two host-skipped suites run in the openpi venv |
| gpu | `test_gpu.py` | built images + GPU | JAX compiles for the card with its bundled ptxas, both venvs see the GPU, and a real checkpoint is served without the config registry |
| check | `test_check.py` | `uv` + images + GPU; `--hf` cases need network | `check.py` run as a contestant does, against the built images: the pass case, each broken adapter, the real checkpoints |
| hf | `test_hf_models.py` | images + GPU + fetched models | Hub checkpoints inside the image: the YUBI-shaped SmolVLA and GR00T fixtures serve, `lerobot/smolvla_base` is refused at load, the pi0.5 port loads |

```sh
cd evaluation-runner
python3 -m pytest tests/ -q                 # everything available here
python3 -m pytest tests/ -q -m "not slow"   # skip slow container runs and model loads
python3 -m pytest tests/ -q -rs             # explain what was skipped and why
```

Overrides: `UMI_ARENA_IMAGE_OPENPI` (default `umi-arena-openpi:dev`),
`UMI_ARENA_IMAGE_TORCH` (default `umi-arena-torch:dev`), `UMI_ARENA_TEST_CHECKPOINT`
(a pi0.5 submission directory), `UMI_ARENA_HF_DIR` (default `~/umi-arena-hf`, the
fixtures and the Hub cache for `test_check.py` and `test_hf_models.py`),
`UMI_ARENA_TEST_TMP` (a directory the Docker daemon can bind-mount).

`test_replay.py` checks dataset alignment, pose integration, sampling rates,
packed-video timestamps, scoring, and replay through the WebSocket server.
It also checks suite membership, subtask weighting and policy cleanup between recordings.
It needs SciPy, PyArrow, PyAV, and Pillow. Run it with:

```sh
uv run --with pytest --with scipy --with pyarrow --with av --with pillow --with websockets \
  --with 'openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client' \
  python -m pytest tests/test_replay.py -q
```
