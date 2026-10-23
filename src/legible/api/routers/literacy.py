"""Phase 2: reading and spelling flash cards (dyslexia).

``GET /session`` builds a session for the learner's current phonics pattern plus spaced
review and stores it (with the answers) in a ``PracticeSession``. Each answer is checked
and explained immediately and updates the word's memory state; ``/complete`` scores
reading and spelling separately and moves each track once.
"""

from __future__ import annotations

import random
from datetime import timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ...literacy import READING, SPELLING, CardPlan, MemoryState, build_session, check_answer
from ...literacy import update as update_memory
from ..db import utcnow
from ..deps import AppSettings, CurrentUser, DbSession, Today
from ..models import PracticeSession, User, WordMemory
from ..progress import get_level, record_session, snapshot
from ..schemas import ProgressionUpdateOut, TrackSnapshot
from ..services.handwriting import ModelUnavailable
from .handwriting import read_image_upload

router = APIRouter(prefix="/api/literacy", tags=["literacy"])

SESSION_KIND = "literacy"


class LiteracyHintOut(BaseModel):
    length: int
    first_letter: str | None


class LiteracyCardOut(BaseModel):
    card_id: str
    word: str
    mode: Literal["read", "spell"]
    skill: str
    skill_label: str
    graphemes: list[str]
    focus: tuple[int, int] | None
    prompt: str
    choices: list[str] | None
    sentence: str | None
    exposure_ms: int | None
    hint: LiteracyHintOut | None


class LiteracySessionOut(BaseModel):
    session_id: str
    skill: str
    skill_label: str
    tracks: list[TrackSnapshot]
    cards: list[LiteracyCardOut]


class LiteracyAnswerIn(BaseModel):
    card_id: str = Field(max_length=32)
    response: str = Field(max_length=100)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class LiteracyErrorOut(BaseModel):
    type: Literal["reversal", "transposition", "omission", "insertion", "substitution", "vowel", "doubling"]
    position: int
    detail: str


class LiteracyAnswerOut(BaseModel):
    correct: bool
    expected: str
    phonetically_plausible: bool
    feedback: str
    errors: list[LiteracyErrorOut]


class LiteracyHandwritingAnswerOut(LiteracyAnswerOut):
    recognized: str


class LiteracyCompletionOut(BaseModel):
    score: float
    perfect: bool
    correct: int
    total: int
    progressions: list[ProgressionUpdateOut]


# --------------------------------------------------------------------------- endpoints


@router.get("/session", response_model=LiteracySessionOut)
def new_session(user: CurrentUser, db: DbSession) -> LiteracySessionOut:
    reading_level = get_level(db, user.id, READING)
    spelling_level = get_level(db, user.id, SPELLING)
    memory = {(row.word, row.mode): _state(row)
              for row in db.scalars(select(WordMemory).where(WordMemory.user_id == user.id))}
    plan = build_session(reading_level, spelling_level, memory, utcnow(), random.Random())
    session = PracticeSession(
        user_id=user.id,
        kind=SESSION_KIND,
        payload={"skill": plan.skill.id, "cards": [card.public() for card in plan.cards]},
        responses={},
    )
    db.add(session)
    db.commit()
    return LiteracySessionOut(
        session_id=session.id,
        skill=plan.skill.id,
        skill_label=plan.skill.label,
        tracks=[snapshot(db, user.id, READING), snapshot(db, user.id, SPELLING)],
        cards=[LiteracyCardOut.model_validate(card.public()) for card in plan.cards],
    )


@router.post("/session/{session_id}/answer", response_model=LiteracyAnswerOut)
def answer(session_id: str, body: LiteracyAnswerIn, user: CurrentUser, db: DbSession) -> LiteracyAnswerOut:
    return LiteracyAnswerOut(**_answer(db, user, session_id, body.card_id, body.response, body.latency_ms))


@router.post("/session/{session_id}/answer-handwriting", response_model=LiteracyHandwritingAnswerOut)
async def answer_handwriting(
    session_id: str,
    request: Request,
    user: CurrentUser,
    settings: AppSettings,
    file: Annotated[UploadFile, File()],
    card_id: Annotated[str, Form(max_length=32)],
    latency_ms: Annotated[float, Form(ge=0, allow_inf_nan=False)] = 0.0,
) -> LiteracyHandwritingAnswerOut:
    """A spelling answer written on paper: the handwriting model reads it, then it is checked."""
    data = await read_image_upload(file, settings)
    with request.app.state.sessionmaker() as db:  # validate the card before the (slower) model call
        _card_for_answer(_owned_session(db, user.id, session_id), card_id)
    try:
        recognized = await request.app.state.handwriting.read_word(data, user_id=user.id)
    except ModelUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    def save() -> dict[str, Any]:
        with request.app.state.sessionmaker() as db:
            owner = db.get(User, user.id)
            return _answer(db, owner, session_id, card_id, recognized, latency_ms)

    result = await run_in_threadpool(save)
    return LiteracyHandwritingAnswerOut(**result, recognized=recognized)


@router.post("/session/{session_id}/complete", response_model=LiteracyCompletionOut)
def complete(session_id: str, user: CurrentUser, db: DbSession, today: Today) -> LiteracyCompletionOut:
    session = _owned_session(db, user.id, session_id)
    responses: dict[str, Any] = session.responses or {}
    if not responses:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Answer at least one card before finishing.")
    # Claim the session atomically, so a double submit can never move a track twice.
    claimed = db.execute(
        update(PracticeSession)
        .where(PracticeSession.id == session.id, PracticeSession.completed_at.is_(None))
        .values(completed_at=utcnow())
    )
    if claimed.rowcount != 1:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This session is already finished.")

    cards = [CardPlan.from_dict(card) for card in session.payload["cards"]]
    results = {mode: [bool(responses.get(c.card_id, {}).get("correct")) for c in cards if c.mode == mode]
               for mode in ("read", "spell")}
    progressions = []
    for mode, track in (("read", READING), ("spell", SPELLING)):
        if results[mode]:
            score = sum(results[mode]) / len(results[mode])
            progressions.append(record_session(db, user, track, score, today))
    correct = sum(results["read"]) + sum(results["spell"])
    total = len(cards)
    outcome = LiteracyCompletionOut(score=round(correct / total, 4), perfect=correct == total, correct=correct,
                                    total=total, progressions=progressions)
    session.result = outcome.model_dump(mode="json")
    db.commit()
    return outcome


# --------------------------------------------------------------------------- helpers


def _owned_session(db: Session, user_id: str, session_id: str) -> PracticeSession:
    session = db.get(PracticeSession, session_id)
    if session is None or session.user_id != user_id or session.kind != SESSION_KIND:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="We couldn’t find that session. Please start a new one.")
    return session


def _card_for_answer(session: PracticeSession, card_id: str) -> CardPlan:
    if session.completed_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This session is already finished.")
    card = next((c for c in session.payload["cards"] if c["card_id"] == card_id), None)
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That card isn’t part of this session.")
    if card_id in (session.responses or {}):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="That card was already answered.")
    return CardPlan.from_dict(card)


def _answer(db: Session, user: User, session_id: str, card_id: str, response: str, latency_ms: float) -> dict:
    session = _owned_session(db, user.id, session_id)
    card = _card_for_answer(session, card_id)
    check = check_answer(card, response)
    now = utcnow()
    session.responses = {
        **(session.responses or {}),
        card_id: {"response": response, "correct": check.correct, "latency_ms": latency_ms,
                  "answered_at": now.isoformat()},
    }
    row = db.scalars(select(WordMemory).where(WordMemory.user_id == user.id, WordMemory.word == card.word,
                                              WordMemory.mode == card.mode)).first()
    if row is None:
        row = WordMemory(user_id=user.id, word=card.word, mode=card.mode, skill=card.skill, seen=0, correct=0,
                         wrong=0, half_life_days=1.0)
        db.add(row)
    state = update_memory(_state(row), check.correct, latency_ms, now)
    row.seen, row.correct, row.wrong = state.seen, state.correct, state.wrong
    row.half_life_days, row.last_seen_at, row.skill = state.half_life_days, state.last_seen_at, card.skill
    db.commit()
    return {"correct": check.correct, "expected": check.expected,
            "phonetically_plausible": check.phonetically_plausible, "feedback": check.feedback,
            "errors": check.errors}


def _state(row: WordMemory) -> MemoryState:
    last = row.last_seen_at
    if last is not None and last.tzinfo is None:  # SQLite returns naive datetimes
        last = last.replace(tzinfo=timezone.utc)
    return MemoryState(seen=row.seen or 0, correct=row.correct or 0, wrong=row.wrong or 0,
                       half_life_days=row.half_life_days or 1.0, last_seen_at=last)

