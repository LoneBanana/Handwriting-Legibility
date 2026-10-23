"""Phase 4 conversation practice: ask the avatar anything, then check what was understood.

* ``POST /sessions`` reads the learner's two converse tracks (speaking speed, captions,
  expression strength, check size) and, if asked, starts a Tavus avatar, falling back
  to the built-in avatar whenever Tavus is not configured or fails.
* ``POST /sessions/{id}/messages`` streams the answer as server-sent events: one
  ``segment`` per sentence as soon as it is complete, so the avatar can start speaking
  straight away, then ``done`` with the full text and the Tavus ``echo_text`` (or
  ``error``). Claude writes the answer; without credentials the offline demo guide does.
* ``POST /sessions/{id}/turns/{turn_id}/check`` and ``POST /checks/{id}/submit`` ask and
  score a few questions about one answer. Scoring is what moves the two tracks.

Database work never blocks the event loop: plain ``def`` endpoints already run in the
thread pool, and async code hands it to ``run_in_threadpool`` with its own session.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Annotated, Any, AsyncIterator, Callable, Literal, Sequence, TypeVar, cast

import anyio
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from ...converse.checks import Question, build_questions, score_check
from ...converse.demo import stream_demo_answer
from ...converse.prompts import GUIDE_SYSTEM_PROMPT, answer_messages
from ...converse.segments import Segment, SegmentParser, echo_text, plain_text, tagged_text
from ...converse.tavus import TavusClient, TavusConversation
from ...progression import caption_mode
from ..db import new_id, utcnow
from ..deps import CurrentUser, DbSession, Today
from ..models import ComprehensionCheckRow, ConverseSessionRow, ConverseTurn
from ..progress import record_session, snapshot
from ..ratelimit import llm_limiter
from ..schemas import ProgressionUpdateOut, TrackSnapshot
from ..services.llm import LLM, LLMError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/converse", tags=["converse"])

T = TypeVar("T")
CaptionMode = Literal["full", "delayed", "off"]

LISTENING = "converse.listening"
EXPRESSIONS = "converse.expressions"
HISTORY_TURNS = 6          # earlier turns replayed to Claude so follow-up questions work
ANSWER_MAX_TOKENS = 2048
ANSWER_LIMIT_MESSAGE = "You’ve asked lots of questions this hour. Let’s take a short break and try again soon."
CHECK_LIMIT_MESSAGE = "That’s lots of practice for one hour. Let’s take a short break and try the questions again soon."


# ---------------------------------------------------------------------------- schemas


class SessionIn(BaseModel):
    provider: Literal["local", "tavus"] = "local"


class TavusOut(BaseModel):
    conversation_id: str
    conversation_url: str
    meeting_token: str | None = None


class ConverseSessionOut(BaseModel):
    session_id: str
    provider: Literal["local", "tavus"]
    assistant: Literal["claude", "demo"]
    words_per_minute: float
    expression_intensity: float
    captions: CaptionMode
    tracks: list[TrackSnapshot]
    tavus: TavusOut | None


class MessageIn(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class CheckQuestionOut(BaseModel):
    question_id: str
    kind: Literal["main_idea", "detail", "emotion", "inference"]
    prompt: str
    choices: list[str]


class CheckOut(BaseModel):
    check_id: str
    questions: list[CheckQuestionOut]


class AnswerIn(BaseModel):
    question_id: str = Field(max_length=64)
    choice: str = Field(max_length=500)


class SubmitIn(BaseModel):
    answers: list[AnswerIn] = Field(max_length=20)


class QuestionResultOut(BaseModel):
    question_id: str
    correct: bool
    expected: str
    explanation: str


class CheckResultOut(BaseModel):
    score: float
    perfect: bool
    results: list[QuestionResultOut]
    progressions: list[ProgressionUpdateOut]


class OkOut(BaseModel):
    ok: Literal[True] = True


# ---------------------------------------------------------------------------- sessions


@dataclass(frozen=True)
class _Plan:
    """The learner's current converse settings, read from their two tracks."""

    tracks: list[TrackSnapshot]
    words_per_minute: float
    expression_intensity: float  # 0.3..1.0
    captions: CaptionMode
    questions: int
    emotion_choices: int

    def stored(self) -> dict[str, Any]:
        return {
            "words_per_minute": self.words_per_minute,
            "expression_intensity": self.expression_intensity,
            "captions": self.captions,
            "questions": self.questions,
            "emotion_choices": self.emotion_choices,
            "levels": {snap.track: snap.level for snap in self.tracks},
        }


def _plan(db: Session, user_id: str) -> _Plan:
    listening = snapshot(db, user_id, LISTENING)
    expressions = snapshot(db, user_id, EXPRESSIONS)
    listen = {name: parameter.value for name, parameter in listening.parameters.items()}
    express = {name: parameter.value for name, parameter in expressions.parameters.items()}
    return _Plan(
        tracks=[listening, expressions],
        words_per_minute=round(listen["words_per_minute"], 1),
        expression_intensity=round(express["expression_intensity"] / 100, 3),
        captions=cast(CaptionMode, caption_mode(listen["caption_support"])),
        questions=int(listen["questions"]),
        emotion_choices=int(express["emotion_choices"]),
    )


@router.post("/sessions", response_model=ConverseSessionOut)
async def create_session(request: Request, user: CurrentUser, body: SessionIn | None = None) -> ConverseSessionOut:
    state = request.app.state
    user_id = user.id
    plan = await run_in_threadpool(_run_db, state.sessionmaker, _plan, user_id)
    session_id = new_id()
    tavus = await _start_tavus(request, session_id) if body is not None and body.provider == "tavus" else None
    assistant: Literal["claude", "demo"] = "claude" if state.llm.available else "demo"
    row = ConverseSessionRow(
        id=session_id,
        user_id=user_id,
        provider="tavus" if tavus else "local",
        settings={**plan.stored(), "assistant": assistant},
        tavus_conversation_id=tavus.conversation_id if tavus else None,
    )
    try:
        await run_in_threadpool(_run_db, state.sessionmaker, _insert, row)
    except Exception:
        if tavus is not None:
            await _end_tavus(request, tavus.conversation_id)
        raise
    return ConverseSessionOut(
        session_id=session_id,
        provider="tavus" if tavus else "local",
        assistant=assistant,
        words_per_minute=plan.words_per_minute,
        expression_intensity=plan.expression_intensity,
        captions=plan.captions,
        tracks=plan.tracks,
        tavus=TavusOut(
            conversation_id=tavus.conversation_id,
            conversation_url=tavus.conversation_url,
            meeting_token=tavus.meeting_token,
        )
        if tavus
        else None,
    )


@router.post("/sessions/{session_id}/end", response_model=OkOut)
async def end_session(session_id: str, request: Request, user: CurrentUser) -> OkOut:
    """Ends the conversation (idempotent) and, best effort, its Tavus call."""
    conversation_id = await run_in_threadpool(_run_db, request.app.state.sessionmaker, _mark_ended, user.id, session_id)
    if conversation_id:
        await _end_tavus(request, conversation_id)
    return OkOut()


def _mark_ended(db: Session, user_id: str, session_id: str) -> str | None:
    """Sets ``ended_at``; returns the Tavus conversation to end, if this call ended the session."""
    session = _owned_session(db, user_id, session_id)
    if session.ended_at is not None:
        return None
    session.ended_at = utcnow()
    db.commit()
    return session.tavus_conversation_id


def _tavus_client(request: Request) -> TavusClient | None:
    """``app.state.tavus`` when one is attached (e.g. by tests), else one built from settings."""
    client = getattr(request.app.state, "tavus", None)
    return client if client is not None else TavusClient.from_settings(request.app.state.settings)


async def _start_tavus(request: Request, session_id: str) -> TavusConversation | None:
    client = _tavus_client(request)
    if client is None:
        log.info("A Tavus avatar was requested but Tavus is not configured; using the built-in avatar")
        return None
    try:
        return await client.create_conversation(f"Legible talk {session_id[:8]}")
    except Exception as exc:  # any failure: the built-in avatar works without Tavus
        log.warning("Could not start a Tavus conversation (%s); using the built-in avatar", exc)
        return None


async def _end_tavus(request: Request, conversation_id: str) -> None:
    client = _tavus_client(request)
    if client is None:
        return
    try:
        await client.end_conversation(conversation_id)
    except Exception as exc:  # Tavus also ends idle calls itself (participant timeouts)
        log.warning("Could not end Tavus conversation %s (%s)", conversation_id, exc)


# ---------------------------------------------------------------------------- messages


@router.post(
    "/sessions/{session_id}/messages",
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "segment events, then done or error"}},
)
def send_message(
    session_id: str, body: MessageIn, request: Request, user: CurrentUser, db: DbSession
) -> EventSourceResponse:
    session = _owned_session(db, user.id, session_id)
    if session.ended_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This conversation has ended. Start a new one to keep talking.")
    state = request.app.state
    if state.llm.available:  # a plain 429 before streaming starts; the offline demo is not limited
        llm_limiter.hit(user.id, ANSWER_LIMIT_MESSAGE)
    history = _history(db, session.id)
    turn = ConverseTurn(session_id=session.id, user_text=body.text)
    db.add(turn)
    db.commit()
    events = _answer_events(state.llm, state.sessionmaker, turn.id, body.text, history)
    return EventSourceResponse(events, sep="\n")


def _history(db: Session, session_id: str) -> list[tuple[str, str]]:
    """The session's latest finished turns, oldest first, as (question, tagged answer)."""
    turns = db.scalars(
        select(ConverseTurn)
        .where(ConverseTurn.session_id == session_id, ConverseTurn.complete.is_(True))
        .order_by(ConverseTurn.created_at.desc())
        .limit(HISTORY_TURNS)
    ).all()
    return [(turn.user_text, tagged_text(_segments(turn))) for turn in reversed(turns)]


async def _answer_events(
    llm: LLM, maker: sessionmaker, turn_id: str, question: str, history: list[tuple[str, str]]
) -> AsyncIterator[dict[str, str]]:
    parser = SegmentParser()
    segments: list[Segment] = []
    chunks = _answer_chunks(llm, question, history)
    try:
        try:
            async for chunk in chunks:
                for segment in parser.feed(chunk):
                    segments.append(segment)
                    yield _segment_event(turn_id, segment)
        finally:
            await _aclose(chunks)
        for segment in parser.finish():
            segments.append(segment)
            yield _segment_event(turn_id, segment)
    except LLMError as exc:
        yield _event("error", {"detail": str(exc)})
        return
    except Exception:
        log.exception("Answering turn %s failed", turn_id)
        yield _event("error", {"detail": "Something went wrong while answering. Please try again."})
        return

    if not segments:
        yield _event("error", {"detail": "The guide didn’t find an answer that time. Please try asking again."})
        return
    try:
        await run_in_threadpool(_run_db, maker, _complete_turn, turn_id, segments)
    except Exception:
        log.exception("Saving turn %s failed", turn_id)
        yield _event("error", {"detail": "The answer couldn’t be saved. Please try again."})
        return
    yield _event("done", {"turn_id": turn_id, "text": plain_text(segments), "echo_text": echo_text(segments)})


def _answer_chunks(llm: LLM, question: str, history: list[tuple[str, str]]) -> AsyncIterator[str]:
    if llm.available:
        return llm.stream_text(
            system=GUIDE_SYSTEM_PROMPT,
            messages=answer_messages(history, question),
            max_tokens=ANSWER_MAX_TOKENS,
            effort="low",  # short, simple answers: speed matters more than deliberation
        )
    return stream_demo_answer(question)


async def _aclose(iterator: Any) -> None:
    """Closes the answer stream (and Claude's HTTP response) even while being cancelled."""
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        with anyio.CancelScope(shield=True):
            await aclose()


def _complete_turn(db: Session, turn_id: str, segments: list[Segment]) -> None:
    turn = db.get(ConverseTurn, turn_id)
    if turn is None:  # the account was deleted while the answer streamed
        return
    turn.assistant_text = plain_text(segments)
    turn.segments = [segment.as_dict() for segment in segments]
    turn.complete = True
    db.commit()


def _segment_event(turn_id: str, segment: Segment) -> dict[str, str]:
    return _event("segment", {"turn_id": turn_id, **segment.as_dict()})


def _event(name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}


# ---------------------------------------------------------------------------- checks


@dataclass(frozen=True)
class _StoredCheck:
    check_id: str
    questions: list[Question]

    def out(self) -> CheckOut:
        return CheckOut(
            check_id=self.check_id,
            questions=[CheckQuestionOut(**question.public()) for question in self.questions],
        )


@dataclass(frozen=True)
class _CheckContext:
    user_text: str
    segments: list[Segment]
    heard: list[str]           # what the avatar said in the session's other answers
    questions: int
    emotion_choices: int
    existing: _StoredCheck | None


@router.post("/sessions/{session_id}/turns/{turn_id}/check", response_model=CheckOut)
async def create_check(session_id: str, turn_id: str, request: Request, user: CurrentUser) -> CheckOut:
    """Questions about one answer. Asking again before submitting returns the same check."""
    maker = request.app.state.sessionmaker
    user_id = user.id
    context = await run_in_threadpool(_run_db, maker, _check_context, user_id, session_id, turn_id)
    if context.existing is not None:
        return context.existing.out()
    if request.app.state.llm.available and context.questions > 0:  # only Claude-written questions count
        llm_limiter.hit(user_id, CHECK_LIMIT_MESSAGE)
    questions = await build_questions(
        request.app.state.llm,
        seed=turn_id,
        learner_question=context.user_text,
        segments=context.segments,
        listening_count=context.questions,
        emotion_choices=context.emotion_choices,
        heard=context.heard,
    )
    if not questions:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="There isn’t enough in that answer to ask about. Try another question first."
        )
    stored = await run_in_threadpool(_run_db, maker, _store_check, user_id, turn_id, questions)
    return stored.out()


def _check_context(db: Session, user_id: str, session_id: str, turn_id: str) -> _CheckContext:
    session = _owned_session(db, user_id, session_id)
    turn = db.get(ConverseTurn, turn_id)
    if turn is None or turn.session_id != session.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="We couldn’t find that answer.")
    if not turn.complete:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="That answer didn’t finish, so there is nothing to check yet.")
    checks = _checks_for_turn(db, user_id, turn.id)
    if any(check.submitted_at is not None for check in checks):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="You already answered the questions about this answer.")

    others = db.scalars(
        select(ConverseTurn).where(ConverseTurn.session_id == session.id, ConverseTurn.id != turn.id)
    ).all()
    stored = session.settings or {}
    if "questions" not in stored or "emotion_choices" not in stored:
        stored = {**_plan(db, user_id).stored(), **stored}
    return _CheckContext(
        user_text=turn.user_text,
        segments=_segments(turn),
        heard=[segment.text for other in others for segment in _segments(other)],
        questions=int(stored["questions"]),
        emotion_choices=int(stored["emotion_choices"]),
        existing=_stored_check(checks[0]) if checks else None,
    )


def _store_check(db: Session, user_id: str, turn_id: str, questions: list[Question]) -> _StoredCheck:
    # A second request for the same turn may have finished first; reuse its check.
    existing = _checks_for_turn(db, user_id, turn_id)
    if existing:
        return _stored_check(existing[0])
    row = ComprehensionCheckRow(
        user_id=user_id, turn_id=turn_id, questions=[question.to_dict() for question in questions]
    )
    db.add(row)
    db.commit()
    return _StoredCheck(check_id=row.id, questions=questions)


def _checks_for_turn(db: Session, user_id: str, turn_id: str) -> Sequence[ComprehensionCheckRow]:
    return db.scalars(
        select(ComprehensionCheckRow)
        .where(ComprehensionCheckRow.turn_id == turn_id, ComprehensionCheckRow.user_id == user_id)
        .order_by(ComprehensionCheckRow.created_at)
    ).all()


def _stored_check(row: ComprehensionCheckRow) -> _StoredCheck:
    return _StoredCheck(check_id=row.id, questions=[Question.from_dict(item) for item in row.questions or []])


@router.post("/checks/{check_id}/submit", response_model=CheckResultOut)
def submit_check(check_id: str, body: SubmitIn, user: CurrentUser, db: DbSession, today: Today) -> CheckResultOut:
    check = db.get(ComprehensionCheckRow, check_id)
    if check is None or check.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="We couldn’t find that check.")
    # Claim the check atomically, so a double submit can never move a track twice.
    claimed = db.execute(
        update(ComprehensionCheckRow)
        .where(ComprehensionCheckRow.id == check.id, ComprehensionCheckRow.submitted_at.is_(None))
        .values(submitted_at=utcnow())
    )
    if claimed.rowcount != 1:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="These answers were already submitted.")

    answers: dict[str, str] = {}
    for answer in body.answers:
        answers.setdefault(answer.question_id, answer.choice)
    scored = score_check([Question.from_dict(item) for item in check.questions or []], answers)
    progressions: list[ProgressionUpdateOut] = []
    if scored.listening is not None:
        progressions.append(record_session(db, user, LISTENING, scored.listening, today))
    if scored.expressions is not None:
        progressions.append(record_session(db, user, EXPRESSIONS, scored.expressions, today))
    result = CheckResultOut(
        score=round(scored.score, 4),
        perfect=scored.perfect,
        results=[QuestionResultOut(**item) for item in scored.results],
        progressions=progressions,
    )
    check.result = result.model_dump(mode="json")
    db.commit()
    return result


# ---------------------------------------------------------------------------- helpers


def _run_db(maker: sessionmaker, fn: Callable[..., T], *args: Any) -> T:
    """Runs ``fn(db, *args)`` with a fresh session (call it through ``run_in_threadpool``)."""
    with maker() as db:
        return fn(db, *args)


def _insert(db: Session, row: Any) -> None:
    db.add(row)
    db.commit()


def _owned_session(db: Session, user_id: str, session_id: str) -> ConverseSessionRow:
    session = db.get(ConverseSessionRow, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="We couldn’t find that conversation.")
    return session


def _segments(turn: ConverseTurn) -> list[Segment]:
    try:
        return [Segment.from_dict(item) for item in turn.segments or []]
    except (KeyError, TypeError, ValueError):
        log.warning("turn %s has malformed segments", turn.id)
        return []
