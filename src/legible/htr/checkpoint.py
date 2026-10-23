"""Checkpoint files.

* Training checkpoints (``<run>/checkpoints/step_00012000.pt``) hold everything needed
  to resume exactly: weights, EMA weights, optimizer, scheduler, scaler, data position.
* Inference checkpoints (``<run>/best.pt``, ``<run>/final.pt``) hold only what serving
  needs: model config, vocabulary, EMA weights, and metadata. Point the API's
  ``LEGIBLE_HTR_CHECKPOINT`` at one of these.

All writes are atomic (write to a temp file, then rename) so a job killed mid-save
never leaves a truncated checkpoint behind.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, from_dict
from .model import HybridHTR
from .vocab import Vocab

log = logging.getLogger(__name__)

FORMAT = "legible-htr"
FORMAT_VERSION = 1
_STEP_FILE = re.compile(r"step_(\d+)\.pt$")


def save_atomic(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def inference_payload(model_cfg: ModelConfig, vocab: Vocab, state_dict: dict[str, torch.Tensor],
                      meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_config": dataclasses.asdict(model_cfg),
        "vocab": vocab.chars,
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "meta": meta,
    }


def read_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_config" not in payload or "vocab" not in payload:
        raise ValueError(f"{path} is not a Legible handwriting checkpoint")
    return payload


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[HybridHTR, Vocab, dict[str, Any]]:
    """Builds the model described by a checkpoint (inference or training) and loads its weights.

    Training checkpoints load their EMA weights, which is what evaluation and serving use.
    """
    payload = read_payload(path)
    cfg = from_dict(ModelConfig, payload["model_config"])
    vocab = Vocab(payload["vocab"])
    model = HybridHTR(cfg, len(vocab))
    state = payload.get("ema") or payload["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, vocab, dict(payload.get("meta") or {})


def load_matching_weights(model: torch.nn.Module, path: str | Path) -> list[str]:
    """Warm-starts ``model`` from a checkpoint, skipping tensors whose shapes differ
    (e.g. a different vocabulary). Returns the names that were not loaded."""
    payload = read_payload(path)
    source = payload.get("ema") or payload["state_dict"]
    target = model.state_dict()
    loadable = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    model.load_state_dict(loadable, strict=False)
    skipped = sorted(set(target) - set(loadable))
    if skipped:
        log.info("warm start from %s: %d tensors not loaded (%s%s)", path, len(skipped),
                 ", ".join(skipped[:5]), "…" if len(skipped) > 5 else "")
    return skipped


def latest_training_checkpoint(run_dir: str | Path) -> Path | None:
    folder = Path(run_dir) / "checkpoints"
    if not folder.is_dir():
        return None
    candidates = [(int(m.group(1)), p) for p in folder.glob("step_*.pt") if (m := _STEP_FILE.search(p.name))]
    return max(candidates)[1] if candidates else None


def prune_training_checkpoints(run_dir: str | Path, keep: int) -> None:
    folder = Path(run_dir) / "checkpoints"
    candidates = sorted((int(m.group(1)), p) for p in folder.glob("step_*.pt") if (m := _STEP_FILE.search(p.name)))
    for _, path in candidates[: max(len(candidates) - keep, 0)]:
        path.unlink(missing_ok=True)
