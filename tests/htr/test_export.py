import pytest

from legible.htr.checkpoint import inference_payload, save_atomic
from legible.htr.export import export_onnx

from .helpers import tiny_model

ort = pytest.importorskip("onnxruntime")


def test_onnx_export_matches_pytorch_at_new_shapes(tmp_path):
    model, vocab = tiny_model(seed=4)
    checkpoint = tmp_path / "tiny.pt"
    save_atomic(inference_payload(model.cfg, vocab, model.state_dict(), {"version": "t"}), checkpoint)
    path = export_onnx(checkpoint, tmp_path / "model.onnx")  # verify() raises on any mismatch
    assert path.stat().st_size > 0


def test_fast_path_uses_onnx_runtime_and_matches_pytorch(tmp_path):
    import io

    import torch
    from PIL import Image

    from legible.htr.inference import Transcriber
    from legible.htr.lora import lora_state

    from .helpers import synthetic_page

    model, vocab = tiny_model(seed=5)
    checkpoint = tmp_path / "tiny.pt"
    save_atomic(inference_payload(model.cfg, vocab, model.state_dict(), {"version": "v-onnx"}), checkpoint)
    onnx_path = export_onnx(checkpoint, tmp_path / "tiny.onnx")
    buffer = io.BytesIO()
    Image.fromarray(synthetic_page(["first line here", "second one"], seed=3)).save(buffer, format="PNG")
    page = buffer.getvalue()

    plain = Transcriber.from_checkpoint(checkpoint, device="cpu")
    fast = Transcriber.from_checkpoint(checkpoint, device="cpu", onnx_path=onnx_path)
    assert plain.onnx is None and fast.onnx is not None
    expected = plain.transcribe(page, mode="page", decoder="fast")
    got = fast.transcribe(page, mode="page", decoder="fast")
    assert [l.text for l in got.lines] == [l.text for l in expected.lines]
    for a, b in zip(got.lines, expected.lines):
        assert [c for c, _ in a.chars] == [c for c, _ in b.chars]
        assert all(abs(p - q) < 1e-3 for (_, p), (_, q) in zip(a.chars, b.chars))

    # Personal adapters only exist in PyTorch, so a personalized request must bypass ONNX.
    def refuse(*_args, **_kwargs):
        raise AssertionError("ONNX must not run while an adapter is active")

    fast.onnx.run = refuse
    state = {k: torch.randn_like(v) * 0.1 for k, v in lora_state(fast.model).items()}
    fast.transcribe(page, mode="page", decoder="fast", adapter={"base_model_version": "v-onnx", "state": state})

    # An export from a different model is never paired with this one.
    other, _ = tiny_model(seed=6)
    other_ckpt = tmp_path / "other.pt"
    save_atomic(inference_payload(other.cfg, vocab, other.state_dict(), {"version": "v-other"}), other_ckpt)
    assert Transcriber.from_checkpoint(other_ckpt, device="cpu", onnx_path=onnx_path).onnx is None
