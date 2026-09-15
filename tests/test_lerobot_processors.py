"""Load saved processor pipelines through the adapter without model weights."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / "adapters" / "lerobot_hf" / "policy.py"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("step_format", ["absent", "registry", "class"])
def test_saved_processor_device_step(tmp_path, monkeypatch, device, step_format):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    spec = importlib.util.spec_from_file_location("lerobot_adapter_under_test", ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    model = torch.nn.Module()
    model.config = SimpleNamespace()
    monkeypatch.setattr(adapter, "get_policy_class", lambda _: SimpleNamespace(from_pretrained=lambda _: model))
    monkeypatch.setattr(adapter.Policy, "_prewarm", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")

    (tmp_path / "config.json").write_text(json.dumps({
        "type": "smolvla",
        "input_features": {
            "observation.images.left": {"shape": [3, 480, 640]},
            "observation.images.right": {"shape": [3, 480, 640]},
            "observation.state": {"shape": [9]},
        },
        "output_features": {"action": {"shape": [16]}},
    }))
    steps = [] if step_format == "absent" else [DeviceProcessorStep(device="cpu")]
    PolicyProcessorPipeline(steps=steps, name="policy_preprocessor").save_pretrained(tmp_path)
    PolicyProcessorPipeline(steps=[], name="policy_postprocessor").save_pretrained(tmp_path)
    preprocessor_path = tmp_path / "policy_preprocessor.json"
    if step_format == "class":
        config = json.loads(preprocessor_path.read_text())
        step = config["steps"][0]
        del step["registry_name"]
        step["class"] = f"{DeviceProcessorStep.__module__}.DeviceProcessorStep"
        preprocessor_path.write_text(json.dumps(config))
    saved_config = preprocessor_path.read_text()

    policy = adapter.Policy(str(tmp_path))
    input_device = device if step_format == "absent" else "cpu"
    state = torch.ones(1, 9, device=input_device)
    output = policy._pre({"observation.state": state})["observation.state"]
    assert output.device.type == device
    torch.testing.assert_close(output.cpu(), state.cpu())
    assert preprocessor_path.read_text() == saved_config
