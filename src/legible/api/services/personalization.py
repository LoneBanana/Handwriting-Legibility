"""Background training of per-user handwriting adapters from consented corrections.

Jobs run one at a time in a single worker thread so they never starve live
transcription. A user's job is scheduled when they have corrected enough lines since
their last adapter; the adapter is only installed if it beats the shared model on that
user's held-out lines (see :mod:`legible.htr.personalize`).
"""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sqlalchemy import select

from ...htr.vocab import normalize_text
from ..db import utcnow
from ..models import HandwritingSample, PersonalAdapter
from ..settings import Settings
from .handwriting import HandwritingService

log = logging.getLogger(__name__)


def sample_lines(settings: Settings, sample: HandwritingSample) -> list[tuple[np.ndarray, str]]:
    """(line image, corrected text) pairs for one sample, or [] when the correction's line
    count does not match the detected lines (then the lines cannot be paired safely)."""
    if not sample.corrected_text:
        return []
    path = settings.uploads_dir / sample.user_id / f"{sample.id}.npz"
    if not path.exists():
        return []
    with np.load(path) as archive:
        images = [archive[key] for key in sorted(archive.files, key=lambda k: int(k.split("_")[1]))]
    texts = [normalize_text(t) for t in sample.corrected_text.split("\n")]
    texts = [t for t in texts if t]
    if len(texts) != len(images):
        return []
    return list(zip(images, texts))


class PersonalizationWorker:
    def __init__(self, settings: Settings, handwriting: HandwritingService, sessionmaker):
        self.settings = settings
        self.handwriting = handwriting
        self.sessionmaker = sessionmaker
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legible-personalize")
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def _new_lines(self, user_id: str) -> tuple[list[tuple[np.ndarray, str]], list[str], int]:
        """All usable corrected lines, the ids of their samples, and how many are new."""
        with self.sessionmaker() as db:
            samples = db.scalars(
                select(HandwritingSample).where(
                    HandwritingSample.user_id == user_id, HandwritingSample.corrected_text.is_not(None)
                )
            ).all()
            pairs: list[tuple[np.ndarray, str]] = []
            ids: list[str] = []
            new = 0
            for sample in samples:
                lines = sample_lines(self.settings, sample)
                if not lines:
                    continue
                pairs.extend(lines)
                ids.append(sample.id)
                if not sample.used_in_adapter:
                    new += len(lines)
        return pairs, ids, new

    def maybe_schedule(self, user_id: str) -> bool:
        if not self.handwriting.configured:
            return False
        _, _, new = self._new_lines(user_id)
        if new < self.settings.personalize_min_samples:
            return False
        with self._lock:
            if user_id in self._pending:
                return False
            self._pending.add(user_id)
        self._executor.submit(self._run, user_id)
        return True

    def _run(self, user_id: str) -> None:
        try:
            self._train(user_id)
        except Exception:  # noqa: BLE001 - a failed job must never take the worker down
            log.exception("personalization failed for user %s", user_id)
        finally:
            with self._lock:
                self._pending.discard(user_id)

    def _train(self, user_id: str) -> None:
        from ...htr.personalize import save_adapter, train_adapter

        pairs, sample_ids, _ = self._new_lines(user_id)
        if len(pairs) < self.settings.personalize_min_samples:
            return
        transcriber = self.handwriting._get_transcriber()
        with self.handwriting._infer_lock:  # copy only while no request has an adapter loaded
            base = copy.deepcopy(transcriber.model)
        result = train_adapter(
            base, transcriber.vocab, transcriber.model_version, pairs,
            steps=self.settings.personalize_steps, device=transcriber.device, copy_base=False,
        )
        log.info("user %s: adapter CER %.4f -> %.4f (%s)", user_id, result.cer_before, result.cer_after,
                 "kept" if result.accepted else "discarded")
        with self.sessionmaker() as db:
            for sample in db.scalars(select(HandwritingSample).where(HandwritingSample.id.in_(sample_ids))):
                sample.used_in_adapter = True
            if result.accepted:
                path: Path = self.handwriting.adapter_path(user_id)
                save_adapter(path, result.adapter)
                row = db.get(PersonalAdapter, user_id) or PersonalAdapter(user_id=user_id)
                row.path = str(path)
                row.base_model_version = transcriber.model_version
                row.samples = len(sample_ids)
                row.lines = len(pairs)
                row.created_at = utcnow()
                db.add(row)
                self.handwriting.forget_adapter(user_id)
            db.commit()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
