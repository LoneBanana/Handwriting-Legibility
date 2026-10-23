"""Phase 1: handwriting photo -> text, corrections for personalization, and privacy controls."""

from __future__ import annotations

import logging
import shutil
from datetime import date
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from starlette.concurrency import run_in_threadpool

from ...progression import TRACKS
from ..db import utcnow
from ..deps import AppSettings, CurrentUser, DbSession, Today
from ..models import HandwritingSample, PersonalAdapter, User
from ..progress import get_level, record_session
from ..ratelimit import llm_limiter
from ..schemas import ProgressionUpdateOut
from ..services.clarify import clarify as clarify_lines
from ..services.handwriting import ModelUnavailable, Transcription
from ..settings import Settings
from .auth import load_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/handwriting", tags=["handwriting"])

IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
TRACK = "handwriting.legibility"


class CharOut(BaseModel):
    char: str
    confidence: float


class LineOut(BaseModel):
    index: int
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    chars: list[CharOut]


class ChangeOut(BaseModel):
    original: str
    corrected: str
    reason: str


class ClarifiedOut(BaseModel):
    text: str
    changes: list[ChangeOut]
    uncertain: list[str]


class TranscribeOut(BaseModel):
    sample_id: str | None
    text: str
    lines: list[LineOut]
    legibility: float
    clarified: ClarifiedOut | None
    model_version: str
    personalized: bool
    elapsed_ms: int
    progressions: list[ProgressionUpdateOut]


class CorrectionIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class StatusOut(BaseModel):
    model_loaded: bool
    model_version: str
    device: str
    personalized: bool


def form_flag(value: str | None, *, default: bool, name: str) -> bool:
    """Parses a true/false multipart field. (FastAPI turns an *omitted* ``bool`` form
    field into False regardless of its default, so flags are read as text.)"""
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"{name} must be true or false.")


async def read_image_upload(file: UploadFile, settings: Settings) -> bytes:
    """Validates an uploaded photo's type and size (shared with the literacy router)."""
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Please upload a PNG, JPEG, or WebP photo.")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        limit = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=f"That photo is larger than {limit} MB.")
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="The uploaded file is empty.")
    return data


def _store_sample(settings: Settings, db, user: User, data: bytes, content_type: str, result: Transcription) -> str:
    sample = HandwritingSample(
        user_id=user.id,
        image_path="",
        lines=[{"bbox": list(line.bbox), "text": line.text, "confidence": line.confidence} for line in result.lines],
        text=result.text,
        model_version=result.model_version,
    )
    db.add(sample)
    db.flush()
    folder = settings.uploads_dir / user.id
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{sample.id}{IMAGE_TYPES[content_type]}"
    image_path.write_bytes(data)
    if result.line_images:
        np.savez_compressed(folder / f"{sample.id}.npz", *result.line_images)
    sample.image_path = str(image_path.relative_to(settings.uploads_dir))
    return sample.id


def _persist(request: Request, user_id: str, data: bytes, content_type: str, result: Transcription,
             track_progress: bool, today: date) -> tuple[str | None, list[ProgressionUpdateOut]]:
    settings: Settings = request.app.state.settings
    with request.app.state.sessionmaker() as db:
        user = db.get(User, user_id)
        sample_id = None
        if load_settings(user).store_handwriting_samples and result.lines:
            sample_id = _store_sample(settings, db, user, data, content_type, result)
        progressions = []
        if track_progress and result.lines:
            spec = TRACKS[TRACK]
            goal = spec.parameter_values(get_level(db, user.id, TRACK))["goal"] / 100.0
            score = min(1.0, result.legibility / goal)
            progressions.append(record_session(db, user, TRACK, score, today))
        db.commit()
    return sample_id, progressions


@router.post("/transcribe", response_model=TranscribeOut)
async def transcribe(
    request: Request,
    user: CurrentUser,
    today: Today,
    settings: AppSettings,
    file: Annotated[UploadFile, File()],
    mode: Annotated[Literal["auto", "page", "line"], Form()] = "auto",
    clarify: Annotated[str | None, Form()] = None,
    decoder: Annotated[Literal["fast", "accurate"], Form()] = "accurate",
    track_progress: Annotated[str | None, Form()] = None,
) -> TranscribeOut:
    clarify_on = form_flag(clarify, default=False, name="clarify")
    track_on = form_flag(track_progress, default=True, name="track_progress")
    data = await read_image_upload(file, settings)
    try:
        result = await request.app.state.handwriting.transcribe(data, mode=mode, decoder=decoder, user_id=user.id)
    except ModelUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    clarified = None
    if clarify_on and request.app.state.llm.available:
        llm_limiter.hit(user.id, "You’ve used “what was meant” a lot this hour. Please try again later.")
        clarified = await clarify_lines(request.app.state.llm, result.lines)
    sample_id, progressions = await run_in_threadpool(
        _persist, request, user.id, data, file.content_type, result, track_on, today
    )
    return TranscribeOut(
        sample_id=sample_id,
        text=result.text,
        lines=[
            LineOut(index=line.index, text=line.text, confidence=line.confidence, bbox=line.bbox,
                    chars=[CharOut(char=c.char, confidence=c.confidence) for c in line.chars])
            for line in result.lines
        ],
        legibility=result.legibility,
        clarified=ClarifiedOut.model_validate(clarified.model_dump()) if clarified else None,
        model_version=result.model_version,
        personalized=result.personalized,
        elapsed_ms=result.elapsed_ms,
        progressions=progressions,
    )


@router.post("/samples/{sample_id}/correction")
def submit_correction(sample_id: str, body: CorrectionIn, request: Request, user: CurrentUser,
                      db: DbSession) -> dict[str, bool]:
    sample = db.get(HandwritingSample, sample_id)
    if sample is None or sample.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That sample wasn’t found.")
    text = body.text.strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="The correction is empty.")
    sample.corrected_text = text
    sample.corrected_at = utcnow()
    sample.used_in_adapter = False
    db.commit()
    personalizer = getattr(request.app.state, "personalizer", None)
    if personalizer is not None:
        personalizer.maybe_schedule(user.id)
    return {"ok": True}


@router.get("/status", response_model=StatusOut)
def model_status(request: Request, user: CurrentUser) -> StatusOut:
    return StatusOut(**request.app.state.handwriting.status(user.id))


@router.delete("/samples", status_code=status.HTTP_204_NO_CONTENT)
def delete_samples(request: Request, user: CurrentUser, db: DbSession, settings: AppSettings) -> Response:
    """Deletes every stored handwriting photo, correction, and the personal adapter."""
    db.execute(delete(HandwritingSample).where(HandwritingSample.user_id == user.id))
    adapter = db.get(PersonalAdapter, user.id)
    if adapter is not None:
        db.delete(adapter)
    db.commit()
    shutil.rmtree(settings.uploads_dir / user.id, ignore_errors=True)
    shutil.rmtree(settings.adapters_dir / user.id, ignore_errors=True)
    request.app.state.handwriting.forget_adapter(user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/samples/count")
def sample_count(user: CurrentUser, db: DbSession) -> dict[str, int]:
    samples = db.scalars(select(HandwritingSample).where(HandwritingSample.user_id == user.id)).all()
    return {"samples": len(samples), "corrected": sum(1 for s in samples if s.corrected_text)}
