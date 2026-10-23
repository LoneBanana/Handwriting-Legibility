"""Character/word error rates of a checkpoint on a manifest.

    legible-eval --checkpoint runs/htr-large/best.pt --manifest data/iam/test.jsonl --decoder accurate

Prints CER/WER, and with ``--predictions out.jsonl`` writes one line per sample
(reference, prediction, per-line CER) to inspect errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data.dataset import finish_line
from .data.image import flatten_background, load_gray, to_ink
from .data.manifest import read_manifest
from .inference import Transcriber
from .metrics import ErrorCounts, edit_distance
from .vocab import normalize_text

log = logging.getLogger(__name__)


def evaluate_manifest(transcriber: Transcriber, manifest: str | Path, decoder: str = "accurate",
                      limit: int = 0, predictions: str | Path | None = None) -> dict[str, float]:
    path = Path(manifest)
    records = read_manifest(path)
    if limit:
        records = records[:limit]
    counts = ErrorCounts()
    out = Path(predictions).open("w", encoding="utf-8") if predictions else None
    started = time.perf_counter()
    batch_inks: list[np.ndarray] = []
    batch_refs: list[tuple[str, str]] = []

    def flush() -> None:
        hyps = transcriber.recognize(batch_inks, decoder)
        for (image, ref), hyp in zip(batch_refs, hyps):
            pred = transcriber.vocab.decode(hyp.ids)
            counts.add(ref, pred)
            if out:
                cer = edit_distance(ref, pred) / max(len(ref), 1)
                out.write(json.dumps({"image": image, "ref": ref, "pred": pred, "cer": round(cer, 4),
                                      "confidence": round(hyp.confidence, 4)}, ensure_ascii=False) + "\n")
        batch_inks.clear()
        batch_refs.clear()

    try:
        for record in records:
            ink = to_ink(flatten_background(load_gray(record.resolve(path.parent))))
            batch_inks.append(finish_line(ink, None, None, transcriber.height, transcriber.max_width, 16))
            batch_refs.append((record.image, normalize_text(record.text)))
            if len(batch_inks) >= 64:
                flush()
        if batch_inks:
            flush()
    finally:
        if out:
            out.close()
    seconds = time.perf_counter() - started
    return {"cer": counts.cer, "wer": counts.wer, "lines": counts.lines,
            "lines_per_s": counts.lines / max(seconds, 1e-6)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a handwriting checkpoint on a manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--decoder", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="only the first N lines of each manifest")
    parser.add_argument("--predictions", help="write per-line predictions (JSONL) here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    transcriber = Transcriber.from_checkpoint(args.checkpoint, device=args.device, beam_size=args.beam_size)
    torch.set_grad_enabled(False)
    for manifest in args.manifest:
        predictions = None
        if args.predictions:
            base = Path(args.predictions)
            predictions = base if len(args.manifest) == 1 else base.with_name(f"{base.stem}.{Path(manifest).stem}{base.suffix}")
        metrics = evaluate_manifest(transcriber, manifest, args.decoder, args.limit, predictions)
        print(f"{manifest}: CER {metrics['cer']:.4f}  WER {metrics['wer']:.4f}  "
              f"({metrics['lines']} lines, {metrics['lines_per_s']:.1f} lines/s, {args.decoder})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
