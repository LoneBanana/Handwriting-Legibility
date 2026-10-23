"""Reads handwriting from a photo: segmentation, recognition, and per-letter confidence.

    legible-transcribe photo.jpg --checkpoint runs/htr-large/best.pt

Two decoders: ``fast`` (greedy CTC, a single pass) and ``accurate`` (attention beam
search rescored with CTC, which also yields alternative readings). Confidence is the
model's probability for each emitted letter; the page's *legibility* is the mean over
letters, which the app shows as a meter and uses to highlight uncertain letters.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_model
from .data.image import load_gray, resize_to_height
from .decoding import Hypothesis, beam_search, ctc_greedy
from .device import pick_device
from .lora import clear_lora, inject_lora, load_lora_state
from .model import HybridHTR
from .segment import LineCrop, find_lines
from .vocab import SPECIAL_TOKENS, Vocab

log = logging.getLogger(__name__)


@dataclass
class LinePrediction:
    text: str
    confidence: float
    chars: list[tuple[str, float]]
    bbox: tuple[int, int, int, int]
    alternatives: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    lines: list[LinePrediction]
    text: str
    legibility: float
    elapsed_ms: int
    line_images: list[np.ndarray] = field(default_factory=list)  # model-height ink maps


class Transcriber:
    def __init__(self, model: HybridHTR, vocab: Vocab, meta: dict[str, Any], device: torch.device,
                 beam_size: int = 6, max_width: int = 2048, batch_lines: int = 16, lora_rank: int = 8):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.meta = meta
        self.device = device
        self.beam_size = beam_size
        self.height = model.cfg.img_height
        self.max_width = max_width
        self.batch_lines = batch_lines
        self.model_version = str(meta.get("version") or "unversioned")
        # Adapter slots are added once at load time (inactive, so outputs are unchanged);
        # personal adapters are then copied in per request.
        self.lora_targets = tuple(inject_lora(self.model, rank=lora_rank))
        self.lora_rank = lora_rank
        self.model.to(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto", beam_size: int = 6) -> "Transcriber":
        dev = pick_device(device)
        model, vocab, meta = load_model(path, "cpu")
        config = meta.get("config") or {}
        max_width = int((config.get("data") or {}).get("max_width") or 2048)
        return cls(model, vocab, meta, dev, beam_size=beam_size, max_width=max_width)

    # ------------------------------------------------------------------ recognition

    def _autocast(self):
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    def prepare(self, ink: np.ndarray) -> np.ndarray:
        return resize_to_height(ink, self.height, self.max_width)

    @torch.inference_mode()
    def recognize(self, lines: list[np.ndarray], decoder: str = "accurate") -> list[Hypothesis]:
        """Recognizes model-height ink maps, batching lines of similar width."""
        if not lines:
            return []
        order = sorted(range(len(lines)), key=lambda i: lines[i].shape[1])
        results: list[Hypothesis | None] = [None] * len(lines)
        multiple = 32
        for start in range(0, len(order), self.batch_lines):
            idx = order[start : start + self.batch_lines]
            widths = [lines[i].shape[1] for i in idx]
            padded = -(-max(widths) // multiple) * multiple
            batch = torch.zeros(len(idx), 1, self.height, padded)
            for row, i in enumerate(idx):
                batch[row, 0, :, : widths[row]] = torch.from_numpy(lines[i])
            images = batch.to(self.device)
            width_t = torch.tensor(widths, device=self.device)
            with self._autocast():
                enc = self.model.encode(images, width_t)
                if decoder == "fast":
                    hyps = ctc_greedy(self.model.ctc_logits(enc), enc.col_lengths)
                else:
                    hyps = beam_search(self.model, enc, beam_size=self.beam_size, vocab=self.vocab)
            for row, i in enumerate(idx):
                results[i] = hyps[row]
        return [h for h in results if h is not None]

    def _use_adapter(self, adapter: dict | None) -> bool:
        if adapter is None:
            return False
        if adapter.get("base_model_version") != self.model_version:
            log.info("ignoring adapter trained for %s (serving %s)", adapter.get("base_model_version"),
                     self.model_version)
            return False
        load_lora_state(self.model, adapter["state"])
        return True

    # ------------------------------------------------------------------ pages

    def find_lines(self, image: Any, mode: str = "auto") -> tuple[list[LineCrop], list[np.ndarray]]:
        """Segmentation and preprocessing (CPU only, no model): line crops and their
        model-height ink maps. Raises ``ValueError`` if the image is unreadable."""
        try:
            gray = load_gray(image)
        except Exception as exc:  # noqa: BLE001 - PIL raises many types for bad input
            raise ValueError("That file isn’t an image we can read.") from exc
        crops = find_lines(gray, mode)
        return crops, [self.prepare(crop.ink) for crop in crops]

    def read(self, crops: list[LineCrop], inks: list[np.ndarray], decoder: str = "accurate",
             adapter: dict | None = None, started: float | None = None) -> PageResult:
        """Recognizes prepared lines (the part that uses the model and any adapter)."""
        started = time.perf_counter() if started is None else started
        try:
            self._use_adapter(adapter)
            hyps = self.recognize(inks, decoder)
        finally:
            clear_lora(self.model)

        lines = []
        for crop, hyp in zip(crops, hyps):
            # Text and per-letter confidences come from the same ids, so highlighting always lines up.
            chars = [(self.vocab.itos[i], round(float(p), 4)) for i, p in zip(hyp.ids, hyp.char_probs)
                     if i >= len(SPECIAL_TOKENS)]
            text = "".join(c for c, _ in chars)
            alternatives = [alt for alt, _ in hyp.alternatives if alt != text][:4]
            lines.append(LinePrediction(text, round(hyp.confidence, 4), chars, crop.bbox, alternatives))
        probs = [p for line in lines for c, p in line.chars if not c.isspace()]
        legibility = float(np.mean(probs)) if probs else 0.0
        return PageResult(
            lines=lines,
            text="\n".join(line.text for line in lines),
            legibility=round(legibility, 4),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            line_images=inks,
        )

    def transcribe(self, image: Any, mode: str = "auto", decoder: str = "accurate",
                   adapter: dict | None = None) -> PageResult:
        """``image``: bytes, a path, a PIL image, or an array. Raises ``ValueError`` if unreadable."""
        started = time.perf_counter()
        crops, inks = self.find_lines(image, mode)
        return self.read(crops, inks, decoder, adapter, started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe handwriting in an image.")
    parser.add_argument("images", nargs="+", help="image files")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["auto", "page", "line"], default="auto")
    parser.add_argument("--decoder", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true", help="print full results as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    transcriber = Transcriber.from_checkpoint(args.checkpoint, device=args.device)
    for path in args.images:
        result = transcriber.transcribe(Path(path), mode=args.mode, decoder=args.decoder)
        if args.json:
            payload = asdict(result)
            payload.pop("line_images")
            print(json.dumps({"image": path, **payload}, ensure_ascii=False))
        else:
            print(f"== {path}  (legibility {result.legibility:.0%}, {result.elapsed_ms} ms)")
            print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
