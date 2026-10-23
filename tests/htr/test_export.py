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
