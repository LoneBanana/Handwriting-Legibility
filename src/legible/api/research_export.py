"""Exports consented, corrected handwriting as a training manifest for the shared model.

    legible-export-research --out data/consented
    legible-prepare split data/consented/all.jsonl --val 0.1 --test 0.1

Real dysgraphic handwriting is the most valuable data the model can get, and almost
none is public. Learners can opt in, separately from keeping samples for their own
adapter, to share their corrected lines. This tool exports only:

* samples of learners whose ``share_samples_for_research`` setting is on *now*
  (turning it off, deleting samples, or deleting the account stops future exports);
* lines whose correction pairs one-to-one with a detected line.

Writers are pseudonymous: a keyed hash of the account id (keyed with the server's
secret), stable across exports so splits stay writer-disjoint, but not reversible
without the server secret. No names or emails are written. The *text* of a line can
still be personal, so treat every export as research data under the consent's terms.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from ..htr.data.manifest import Record, write_manifest
from .db import make_engine, make_sessionmaker
from .models import HandwritingSample, User
from .routers.auth import load_settings
from .services.personalization import sample_lines
from .settings import Settings, get_settings

log = logging.getLogger(__name__)


def pseudonym(user_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()[:16]


def export(settings: Settings, out: Path, sessionmaker=None) -> dict[str, int]:
    """Writes ``out/<writer>/<line>.png`` and ``out/all.jsonl``; returns counts.
    ``sessionmaker`` defaults to a connection to the configured database."""
    engine = None if sessionmaker is not None else make_engine(settings.resolved_database_url)
    maker = sessionmaker or make_sessionmaker(engine)
    secret = settings.secret_key()
    records: list[Record] = []
    stats: Counter[str] = Counter()
    try:
        with maker() as db:
            consenting = [
                user.id for user in db.scalars(select(User))
                if (prefs := load_settings(user)).share_samples_for_research and prefs.store_handwriting_samples
            ]
            stats["writers"] = len(consenting)
            query = select(HandwritingSample).where(
                HandwritingSample.user_id.in_(consenting), HandwritingSample.corrected_text.is_not(None)
            )
            for sample in db.scalars(query):
                stats["samples"] += 1
                pairs = sample_lines(settings, sample)
                if not pairs:
                    stats["samples_unpaired"] += 1
                    continue
                writer = pseudonym(sample.user_id, secret)
                folder = out / writer
                folder.mkdir(parents=True, exist_ok=True)
                for i, (ink, text) in enumerate(pairs):
                    image = (255.0 * (1.0 - np.clip(ink, 0.0, 1.0))).astype(np.uint8)
                    path = folder / f"{sample.id}_{i}.png"
                    cv2.imwrite(str(path), image)
                    records.append(Record(path.relative_to(out).as_posix(), text, writer, image.shape[1],
                                          image.shape[0]))
                    stats["lines"] += 1
    finally:
        if engine is not None:
            engine.dispose()
    write_manifest(out / "all.jsonl", records)
    return dict(stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export consented, corrected handwriting lines as a manifest.")
    parser.add_argument("--out", type=Path, required=True, help="output folder (images + all.jsonl)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats = export(get_settings(), args.out)
    print(f"exported {stats.get('lines', 0)} lines from {stats.get('samples', 0)} corrected samples "
          f"by {stats.get('writers', 0)} consenting writers to {args.out / 'all.jsonl'}"
          + (f" ({stats['samples_unpaired']} samples skipped: correction line count didn't match)"
             if stats.get("samples_unpaired") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
