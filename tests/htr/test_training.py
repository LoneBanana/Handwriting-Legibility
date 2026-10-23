"""End-to-end trainer runs on CPU with a tiny model and synthetic data."""

import json

import pytest
import torch

from legible.htr.checkpoint import latest_training_checkpoint, load_model, read_payload
from legible.htr.config import load_config
from legible.htr.train import RESUME_MARKER, Runtime, Trainer, lr_factor

TINY_OVERRIDES = [
    "model.cnn_channels=[8,16,24,32]", "model.cnn_depths=[1,1,1,1]", "model.d_model=32", "model.n_heads=2",
    "model.enc_layers=1", "model.dec_layers=1", "model.max_decode_len=40",
    "data.train.0.size=48", "data.val.0.size=8", "data.batch_size=8", "data.num_workers=0",
    "data.max_text_len=40", "train.log_every=2", "train.eval_every=4", "train.save_every=4",
    "train.warmup_steps=2", "train.tensorboard=false",
]


def _trainer(tmp_path, *extra):
    cfg = load_config("configs/htr/tiny.yaml", TINY_OVERRIDES + [f"train.output_dir={tmp_path}", *extra])
    return Trainer(cfg, Runtime(0, 1, 0, torch.device("cpu")))


def test_trains_saves_and_resumes(tmp_path):
    _trainer(tmp_path, "train.max_steps=4").fit()
    assert latest_training_checkpoint(tmp_path).name == "step_00000004.pt"
    for name in ("best.pt", "final.pt", "vocab.json", "config.json", "metrics.jsonl"):
        assert (tmp_path / name).exists(), name
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert any(r["split"] == "val" and "cer_att" in r for r in records)

    resumed = _trainer(tmp_path, "train.max_steps=6")
    resumed.fit()
    assert resumed.step == 6
    payload = read_payload(latest_training_checkpoint(tmp_path))
    assert payload["progress"]["step"] == 6

    model, vocab, meta = load_model(tmp_path / "final.pt")
    assert meta["step"] == 6 and len(vocab) == model.vocab_size and not model.training


def test_time_limit_checkpoints_and_marks_for_requeue(tmp_path):
    trainer = _trainer(tmp_path, "train.max_steps=50", "train.max_runtime_minutes=0.000001")
    trainer.fit()
    assert trainer.step == 1
    assert (tmp_path / RESUME_MARKER).exists()
    assert latest_training_checkpoint(tmp_path).name == "step_00000001.pt"


def test_lr_schedule_warms_up_then_decays():
    values = [lr_factor(s, warmup=10, total=100, min_ratio=0.1) for s in range(100)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert values[50] < values[20] and values[-1] == pytest.approx(0.1, abs=1e-3)
