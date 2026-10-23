import io

import numpy as np
import pytest
import torch
from PIL import Image

from legible.htr.checkpoint import inference_payload, save_atomic
from legible.htr.inference import Transcriber
from legible.htr.lora import inject_lora, lora_modules, lora_state, load_lora_state
from legible.htr.personalize import load_adapter, save_adapter, train_adapter

from .helpers import render_line, synthetic_page, tiny_model


@pytest.fixture
def checkpoint(tmp_path):
    model, vocab = tiny_model(seed=7)
    path = tmp_path / "tiny.pt"
    save_atomic(inference_payload(model.cfg, vocab, model.state_dict(), {"version": "tiny-test"}), path)
    return path


def _png(gray: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(gray).save(buffer, format="PNG")
    return buffer.getvalue()


def test_transcribes_a_page_with_boxes_and_confidences(checkpoint):
    transcriber = Transcriber.from_checkpoint(checkpoint, device="cpu", beam_size=2)
    page = synthetic_page(["first line here", "second line", "third"], seed=1)
    for decoder in ("fast", "accurate"):
        result = transcriber.transcribe(_png(page), mode="page", decoder=decoder)
        assert len(result.lines) == 3 and len(result.line_images) == 3
        assert 0.0 <= result.legibility <= 1.0
        assert result.text == "\n".join(line.text for line in result.lines)
        for line in result.lines:
            assert len(line.chars) == len(line.text.replace("<unk>", ""))
            assert all(0.0 <= p <= 1.0 for _, p in line.chars)


def test_unreadable_bytes_raise_value_error(checkpoint):
    transcriber = Transcriber.from_checkpoint(checkpoint, device="cpu")
    with pytest.raises(ValueError):
        transcriber.transcribe(b"definitely not an image")


def test_adapters_swap_in_and_are_cleared_after_each_call(checkpoint):
    transcriber = Transcriber.from_checkpoint(checkpoint, device="cpu", beam_size=1)
    line = _png(synthetic_page(["adapter test"], seed=5))
    base = transcriber.transcribe(line, mode="line", decoder="fast")

    state = {k: torch.randn_like(v) * 0.5 for k, v in lora_state(transcriber.model).items()}
    adapter = {"base_model_version": transcriber.model_version, "state": state}
    personal = transcriber.transcribe(line, mode="line", decoder="fast", adapter=adapter)
    assert personal.lines[0].chars != base.lines[0].chars or personal.lines[0].text != base.lines[0].text
    assert not any(m.active for m in lora_modules(transcriber.model).values())
    again = transcriber.transcribe(line, mode="line", decoder="fast")
    assert again.lines[0].text == base.lines[0].text and again.lines[0].chars == base.lines[0].chars

    stale = {"base_model_version": "some-other-model", "state": state}
    ignored = transcriber.transcribe(line, mode="line", decoder="fast", adapter=stale)
    assert ignored.lines[0].chars == base.lines[0].chars


def test_zero_adapter_is_the_identity():
    model, _ = tiny_model()
    images = torch.rand(1, 1, 64, 128)
    widths = torch.tensor([128])
    with torch.no_grad():
        before = model.ctc_logits(model.encode(images, widths))
        names = inject_lora(model)
        assert names
        load_lora_state(model, {k: torch.zeros_like(v) if k.endswith("lora_b") else v
                                for k, v in lora_state(model).items()})
        after = model.ctc_logits(model.encode(images, widths))
    torch.testing.assert_close(before, after)


def test_train_adapter_reports_metrics_and_round_trips(tmp_path):
    model, vocab = tiny_model(seed=2)
    from legible.htr.data.image import resize_to_height

    texts = ["cat", "dog", "sun", "map", "hat", "pen"]
    samples = [(resize_to_height(render_line(t, seed=i), 64, 512), t) for i, t in enumerate(texts)]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = train_adapter(model, vocab, "tiny-test", samples, steps=6, batch_size=4, seed=0)
    assert set(result.adapter["state"]) and result.train_lines == len(texts)
    assert 0.0 <= result.cer_before and 0.0 <= result.cer_after
    for k, v in model.state_dict().items():  # the base model is untouched
        torch.testing.assert_close(v, before[k])
    path = tmp_path / "adapter.pt"
    save_adapter(path, result.adapter)
    loaded = load_adapter(path)
    assert loaded["base_model_version"] == "tiny-test" and loaded["state"].keys() == result.adapter["state"].keys()
