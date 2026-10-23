"""Trains a person's handwriting adapter from the lines they corrected in the app.

The adapter is only kept if it actually reads that person's held-out lines better
than the shared model does, so personalization can never make transcription worse.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import AugmentConfig
from .data.augment import Augmenter
from .data.dataset import Collator, finish_line
from .decoding import attention_greedy
from .lora import clear_lora, inject_lora, lora_modules, lora_parameters, lora_state, set_lora_active
from .metrics import ErrorCounts
from .model import HybridHTR
from .vocab import Vocab, normalize_text

log = logging.getLogger(__name__)

ADAPTER_FORMAT = "legible-htr-adapter"


@dataclass
class AdapterResult:
    adapter: dict[str, Any]
    cer_before: float
    cer_after: float
    train_lines: int
    val_lines: int

    @property
    def accepted(self) -> bool:
        return self.cer_after < self.cer_before or (self.cer_before == 0.0 and self.cer_after == 0.0)


def _split(samples: list[tuple[np.ndarray, str]], rng: np.random.Generator):
    """Holds out ~20% (at least 2 lines) for the accept/reject decision when there is enough data."""
    order = rng.permutation(len(samples))
    if len(samples) < 10:
        return [samples[i] for i in order], [samples[i] for i in order]
    n_val = max(2, len(samples) // 5)
    return [samples[i] for i in order[n_val:]], [samples[i] for i in order[:n_val]]


@torch.no_grad()
def _cer(model: HybridHTR, vocab: Vocab, samples: list[tuple[np.ndarray, str]], device: torch.device) -> float:
    model.eval()
    counts = ErrorCounts()
    collate = Collator(model.cfg.width_reduction)
    for start in range(0, len(samples), 16):
        chunk = samples[start : start + 16]
        batch = collate([{"image": ink, "text": text, "ids": vocab.encode(text)} for ink, text in chunk])
        enc = model.encode(batch["images"].to(device), batch["widths"].to(device))
        hyps = attention_greedy(model, enc, max_len=int(batch["ctc_lengths"].max()) + 16)
        for (_, text), hyp in zip(chunk, hyps):
            counts.add(text, vocab.decode(hyp.ids))
    return counts.cer


def train_adapter(base: HybridHTR, vocab: Vocab, base_version: str, samples: list[tuple[np.ndarray, str]], *,
                  rank: int = 8, alpha: float = 16.0, steps: int = 300, lr: float = 2e-3, batch_size: int = 8,
                  device: torch.device | str = "cpu", seed: int = 0, augment_strength: float = 0.3,
                  copy_base: bool = True) -> AdapterResult:
    """``samples``: (model-height ink map, corrected text) pairs. ``base`` is left untouched
    unless ``copy_base=False`` (for callers that already passed a private copy)."""
    device = torch.device(device)
    samples = [(ink, normalize_text(text)) for ink, text in samples if normalize_text(text)]
    if len(samples) < 2:
        raise ValueError("need at least two corrected lines to personalize")
    rng = np.random.default_rng(seed)
    train, val = _split(samples, rng)

    model = (copy.deepcopy(base) if copy_base else base).to(device)
    if not lora_modules(model):
        inject_lora(model, rank=rank, alpha=alpha)
    clear_lora(model)
    torch.manual_seed(seed)
    for module in lora_modules(model).values():
        nn.init.kaiming_uniform_(module.lora_a, a=math.sqrt(5))
    for p in model.parameters():
        p.requires_grad_(False)
    params = lora_parameters(model)
    for p in params:
        p.requires_grad_(True)

    cer_before = _cer(model, vocab, val, device)

    set_lora_active(model, True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))
    )
    augmenter = Augmenter(AugmentConfig(strength=augment_strength, p_paper=0.1, p_jpeg=0.1, p_cutout=0.0))
    collate = Collator(model.cfg.width_reduction)
    height = model.cfg.img_height
    for step in range(steps):
        model.train()
        picks = rng.integers(len(train), size=min(batch_size, len(train)))
        batch_samples = []
        for i in picks:
            ink, text = train[int(i)]
            image = finish_line(ink, rng, augmenter, height, max(ink.shape[1] * 2, 64), 16)
            batch_samples.append({"image": image, "text": text, "ids": vocab.encode(text, normalize=False)})
        batch = collate(batch_samples)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["images"], batch["widths"], batch["tgt_in"])
        loss = model.loss(out, batch)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()

    cer_after = _cer(model, vocab, val, device)
    names = sorted(lora_modules(model))
    adapter = {
        "format": ADAPTER_FORMAT,
        "base_model_version": base_version,
        "rank": rank,
        "alpha": alpha,
        "targets": names,
        "state": lora_state(model),
        "metrics": {"cer_before": cer_before, "cer_after": cer_after, "train_lines": len(train),
                    "val_lines": len(val), "steps": steps},
    }
    log.info("adapter: CER %.4f -> %.4f on %d held-out lines", cer_before, cer_after, len(val))
    return AdapterResult(adapter, cer_before, cer_after, len(train), len(val))


def save_adapter(path: str | Path, adapter: dict[str, Any]) -> None:
    from .checkpoint import save_atomic

    save_atomic(adapter, path)


def load_adapter(path: str | Path) -> dict[str, Any]:
    adapter = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(adapter, dict) or adapter.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"{path} is not a Legible handwriting adapter")
    return adapter
