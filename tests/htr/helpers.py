"""Small builders shared by the handwriting tests (tiny models, synthetic pages)."""

from __future__ import annotations

import numpy as np
import torch

from legible.htr.config import ModelConfig
from legible.htr.data.corpus import TextCorpus
from legible.htr.data.synthetic import HandwritingRenderer, discover_fonts
from legible.htr.model import HybridHTR
from legible.htr.vocab import Vocab

TINY = dict(
    cnn_channels=(8, 16, 24, 32), cnn_depths=(1, 1, 1, 1), d_model=32, n_heads=2,
    enc_layers=1, dec_layers=1, drop_path=0.0, dropout=0.0, max_decode_len=24,
)


def tiny_model(seed: int = 0, **overrides) -> tuple[HybridHTR, Vocab]:
    torch.manual_seed(seed)
    vocab = Vocab()
    model = HybridHTR(ModelConfig(**{**TINY, **overrides}), len(vocab)).eval()
    return model, vocab


def renderer(irregularity: float = 0.3) -> HandwritingRenderer:
    return HandwritingRenderer(discover_fonts(), irregularity=irregularity, p_strike=0.0, p_mirror=0.0)


def render_line(text: str, seed: int = 0) -> np.ndarray:
    ink, _ = renderer().render(text, np.random.default_rng(seed))
    return ink


def synthetic_page(texts: list[str], drift: float = 0.0, seed: int = 0) -> np.ndarray:
    """A uint8 grayscale 'photo' (dark ink on light paper) with one rendered line per text.

    ``drift`` shears each line so it climbs (negative) or falls (positive) across the page,
    the way dysgraphic handwriting often drifts off a straight baseline.
    """
    import cv2

    rng = np.random.default_rng(seed)
    lines = []
    for i, text in enumerate(texts):
        line = render_line(text, seed + i)
        if drift:
            h, w = line.shape
            extra = int(abs(drift) * w) + 1
            offset = extra if drift < 0 else 0
            m = np.float32([[1, 0, 0], [drift, 1, offset]])
            line = cv2.warpAffine(line, m, (w, h + extra))
        lines.append(line)
    gap = 40
    width = max(line.shape[1] for line in lines) + 140
    height = sum(line.shape[0] + gap for line in lines) + 80
    page = np.zeros((height, width), np.float32)
    y = 40
    for line in lines:
        h, w = line.shape
        x = 60 + int(rng.integers(0, 20))
        page[y : y + h, x : x + w] = np.maximum(page[y : y + h, x : x + w], line)
        y += h + gap
    gray = 235 - page * 200 + rng.normal(0, 3, page.shape)
    return np.clip(gray, 0, 255).astype(np.uint8)


def corpus() -> TextCorpus:
    return TextCorpus.builtin()
