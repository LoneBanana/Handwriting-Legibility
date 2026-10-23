"""Serves the Phase 1 handwriting model to the API.

The model loads lazily on first use from ``LEGIBLE_HTR_CHECKPOINT``. Inference runs in
a worker thread behind a lock: a single model instance is shared by every request, and
per-user adapters (personalization) are swapped in for the duration of one call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from starlette.concurrency import run_in_threadpool

from ..settings import Settings

log = logging.getLogger(__name__)

Mode = Literal["auto", "page", "line"]
Decoder = Literal["fast", "accurate"]


class ModelUnavailable(RuntimeError):
    """No handwriting model is configured or it failed to load."""


@dataclass
class CharScore:
    char: str
    confidence: float


@dataclass
class LineResult:
    index: int
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    chars: list[CharScore]
    alternatives: list[str] = field(default_factory=list)


@dataclass
class Transcription:
    text: str
    lines: list[LineResult]
    legibility: float
    model_version: str
    elapsed_ms: int
    line_images: list[Any] = field(default_factory=list)  # ink maps kept for personalization
    personalized: bool = False


class HandwritingService:
    def __init__(self, settings: Settings, max_cached_adapters: int = 32):
        self.settings = settings
        self._transcriber = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._adapters: OrderedDict[str, dict] = OrderedDict()
        self._adapters_lock = threading.Lock()
        self._max_cached_adapters = max_cached_adapters

    # ---------------------------------------------------------------- model loading

    @property
    def configured(self) -> bool:
        return self.settings.htr_checkpoint is not None

    def _get_transcriber(self):
        if self._transcriber is not None:
            return self._transcriber
        if not self.configured:
            raise ModelUnavailable(
                "Reading handwriting isn’t set up on this server yet. "
                "(Administrators: set LEGIBLE_HTR_CHECKPOINT to a trained checkpoint; see docs/TRAINING.md.)"
            )
        with self._load_lock:
            if self._transcriber is None:
                if self._load_error is not None:
                    raise ModelUnavailable(self._load_error)
                try:
                    from ...htr.inference import Transcriber

                    started = time.perf_counter()
                    self._transcriber = Transcriber.from_checkpoint(
                        self.settings.htr_checkpoint,
                        device=self.settings.htr_device,
                        beam_size=self.settings.htr_beam_size,
                    )
                    log.info(
                        "loaded handwriting model %s on %s in %.1fs",
                        self._transcriber.model_version,
                        self._transcriber.device,
                        time.perf_counter() - started,
                    )
                except Exception as exc:  # noqa: BLE001 - surface any load failure as "unavailable"
                    log.exception("failed to load handwriting model from %s", self.settings.htr_checkpoint)
                    self._load_error = f"The handwriting model failed to load: {exc}"
                    raise ModelUnavailable(self._load_error) from exc
        return self._transcriber

    @property
    def model_loaded(self) -> bool:
        return self._transcriber is not None

    def status(self, user_id: str | None = None) -> dict[str, Any]:
        loaded = False
        version = ""
        device = ""
        try:
            transcriber = self._get_transcriber()
            loaded, version, device = True, transcriber.model_version, str(transcriber.device)
        except ModelUnavailable:
            pass
        return {
            "model_loaded": loaded,
            "model_version": version,
            "device": device,
            "personalized": bool(user_id) and self.adapter_path(user_id).exists(),
        }

    # ---------------------------------------------------------------- personalization

    def adapter_path(self, user_id: str) -> Path:
        return self.settings.adapters_dir / user_id / "adapter.pt"

    def _adapter_for(self, user_id: str | None) -> dict | None:
        if not user_id:
            return None
        with self._adapters_lock:
            if user_id in self._adapters:
                self._adapters.move_to_end(user_id)
                return self._adapters[user_id]
        path = self.adapter_path(user_id)
        if not path.exists():
            return None
        from ...htr.personalize import load_adapter

        try:
            adapter = load_adapter(path)
        except Exception:  # noqa: BLE001 - a broken adapter must never block transcription
            log.exception("could not load adapter %s; using the base model", path)
            return None
        with self._adapters_lock:
            self._adapters[user_id] = adapter
            while len(self._adapters) > self._max_cached_adapters:
                self._adapters.popitem(last=False)
        return adapter

    def forget_adapter(self, user_id: str) -> None:
        with self._adapters_lock:
            self._adapters.pop(user_id, None)

    # ---------------------------------------------------------------- inference

    def _transcribe_sync(self, data: bytes, mode: Mode, decoder: Decoder, user_id: str | None) -> Transcription:
        transcriber = self._get_transcriber()
        started = time.perf_counter()
        crops, inks = transcriber.find_lines(data, mode)  # CPU work, safe to run concurrently
        adapter = self._adapter_for(user_id)
        with self._infer_lock:  # one request at a time owns the model (and its adapter slot)
            page = transcriber.read(crops, inks, decoder, adapter, started)
        lines = [
            LineResult(
                index=i,
                text=line.text,
                confidence=line.confidence,
                bbox=tuple(int(v) for v in line.bbox),
                chars=[CharScore(c, p) for c, p in line.chars],
                alternatives=list(line.alternatives),
            )
            for i, line in enumerate(page.lines)
        ]
        return Transcription(
            text=page.text,
            lines=lines,
            legibility=page.legibility,
            model_version=transcriber.model_version,
            elapsed_ms=page.elapsed_ms,
            line_images=page.line_images,
            personalized=adapter is not None,
        )

    async def transcribe(
        self, data: bytes, *, mode: Mode = "auto", decoder: Decoder = "accurate", user_id: str | None = None
    ) -> Transcription:
        """Transcribes a photo. Raises :class:`ModelUnavailable` without a model and
        ``ValueError`` when the bytes are not a readable image."""
        return await run_in_threadpool(self._transcribe_sync, data, mode, decoder, user_id)

    async def read_word(self, data: bytes, *, user_id: str | None = None) -> str:
        """Reads a photo of a single handwritten word or line (literacy spelling answers)."""
        result = await self.transcribe(data, mode="line", decoder="accurate", user_id=user_id)
        return " ".join(line.text for line in result.lines).strip()
