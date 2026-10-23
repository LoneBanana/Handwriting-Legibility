"""Phase 3: math-fact fluency rounds (dyscalculia).

``GET /round`` plans a round for the learner's ``numeracy.fluency`` level (see
:mod:`legible.numeracy.rounds`) and keeps its answer key in a ``PracticeSession``: the
client only ever receives the problems. ``POST /round/{round_id}/submit`` grades every
answer for accuracy and speed, updates the learner's memory of each fact (which steers
review in later rounds), and feeds the round's score to the track once.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ...numeracy import TRACK_KEY, AnswerKey, FactStats, Graded, grade, plan_round, record_attempt, score_round
from ..db import utcnow
from ..deps import CurrentUser, DbSession, Today
from ..models import FactMemory, PracticeSession, User
from ..progress import get_level, parameters_out, record_session, spec_for
from ..schemas import ProgressionUpdateOut, TrackParameterOut

router = APIRouter(prefix="/api/numeracy", tags=["numeracy"])

SESSION_KIND = "numeracy"


# ------------------------------------------------------------------------ schemas


class DotsVisualOut(BaseModel):
    type: Literal["dots"]
    count: int
    layout: Literal["dice", "random", "line"]


class TenFrameVisualOut(BaseModel):
    type: Literal["ten_frame"]
    filled: int
    frames: int


class ArrayVisualOut(BaseModel):
    type: Literal["array"]
    rows: int
    cols: int


class NumberLineVisualOut(BaseModel):
    type: Literal["number_line"]
    min: int
    max: int
    target: int | None


VisualOut = Annotated[
    DotsVisualOut | TenFrameVisualOut | ArrayVisualOut | NumberLineVisualOut,
    Field(discriminator="type"),
]


class ProblemOut(BaseModel):
    problem_id: str
    kind: Literal["subitize", "compare", "bond", "add", "sub", "mul", "div", "number_line"]
    prompt: str
    visual: VisualOut | None
    answer_format: Literal["integer", "choice"]
    choices: list[str] | None


class RoundOut(BaseModel):
    round_id: str
    track: str
    skill: str
    skill_label: str
    level: int
    time_limit_ms: int
    parameters: dict[str, TrackParameterOut]
    problems: list[ProblemOut]


class ProblemAnswerIn(BaseModel):
    # A client that sends a bare number (7 rather than "7") is still understood.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    problem_id: str = Field(max_length=64)
    answer: str = Field(max_length=32)
    # No upper bound: a round left open overnight must still be accepted (it simply counts as slow).
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class RoundAnswersIn(BaseModel):
    answers: list[ProblemAnswerIn] = Field(max_length=100)


class ProblemResultOut(BaseModel):
    problem_id: str
    correct: bool
    expected: str
    fluent: bool
    strategy_hint: str | None


class RoundResultOut(BaseModel):
    score: float
    perfect: bool
    fluent_count: int
    total: int
    results: list[ProblemResultOut]
    progressions: list[ProgressionUpdateOut]


# ---------------------------------------------------------------------- endpoints


@router.get("/round", response_model=RoundOut)
def new_round(user: CurrentUser, db: DbSession) -> RoundOut:
    """A round for the learner's level; answers stay on the server until it is submitted."""
    level = get_level(db, user.id, TRACK_KEY)
    memory = {row.fact: _stats(row) for row in db.scalars(select(FactMemory).where(FactMemory.user_id == user.id))}
    plan = plan_round(level, memory, rng=random.Random(), now=utcnow())
    session = PracticeSession(user_id=user.id, kind=SESSION_KIND, payload=plan.to_payload())
    db.add(session)
    db.commit()
    return RoundOut(
        round_id=session.id,
        track=TRACK_KEY,
        skill=plan.skill.id,
        skill_label=plan.skill.label,
        level=level,
        time_limit_ms=plan.time_limit_ms,
        parameters=parameters_out(spec_for(TRACK_KEY), level),
        problems=[ProblemOut.model_validate(problem.public()) for problem in plan.problems],
    )


@router.post("/round/{round_id}/submit", response_model=RoundResultOut)
def submit_round(round_id: str, body: RoundAnswersIn, user: CurrentUser, db: DbSession, today: Today) -> RoundResultOut:
    """Grades a round once: accuracy and speed per problem, fact memory, and the fluency track."""
    session = db.scalars(
        select(PracticeSession).where(
            PracticeSession.id == round_id,
            PracticeSession.user_id == user.id,
            PracticeSession.kind == SESSION_KIND,
        )
    ).first()
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="We couldn’t find that round. Please start a new one.")
    now = utcnow()
    # Claim the round atomically (also on SQLite, which ignores row locks), so a double
    # submit can never grade it or move the track twice.
    claimed = db.execute(
        update(PracticeSession)
        .where(PracticeSession.id == session.id, PracticeSession.completed_at.is_(None))
        .values(completed_at=now)
    )
    if claimed.rowcount != 1:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This round has already been checked.")

    keys = [AnswerKey.from_payload(problem) for problem in session.payload["problems"]]
    responses = _responses_by_problem(body, {key.problem_id for key in keys})
    time_limit_ms = session.payload["time_limit_ms"]
    graded = [grade(key, *responses.get(key.problem_id, (None, None)), time_limit_ms) for key in keys]
    outcome = score_round(graded)

    _remember(db, user, graded, now)
    progression = record_session(db, user, TRACK_KEY, outcome.score, today)
    result = RoundResultOut(
        score=round(outcome.score, 4),
        perfect=outcome.perfect,
        fluent_count=outcome.fluent,
        total=outcome.total,
        results=[
            ProblemResultOut(
                problem_id=g.problem_id,
                correct=g.correct,
                expected=g.expected,
                fluent=g.fluent,
                strategy_hint=g.strategy_hint,
            )
            for g in graded
        ],
        progressions=[progression],
    )
    session.responses = {"answers": [answer.model_dump() for answer in body.answers]}
    session.result = result.model_dump(mode="json")
    db.commit()
    return result


# ------------------------------------------------------------------------ helpers


def _responses_by_problem(body: RoundAnswersIn, problem_ids: set[str]) -> dict[str, tuple[str, float]]:
    responses: dict[str, tuple[str, float]] = {}
    for answer in body.answers:
        if answer.problem_id not in problem_ids:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Some answers are for problems that aren’t in this round.",
            )
        if answer.problem_id in responses:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Each problem can only be answered once.")
        responses[answer.problem_id] = (answer.answer, answer.latency_ms)
    return responses


def _stats(row: FactMemory) -> FactStats:
    return FactStats(
        seen=row.seen or 0,
        correct=row.correct or 0,
        fluent=row.fluent or 0,
        mean_latency_ms=row.mean_latency_ms,
        last_seen_at=row.last_seen_at,
    )


def _remember(db: Session, user: User, graded: list[Graded], now: datetime) -> None:
    """Folds every answer into the learner's fact memory (one row per fact)."""
    facts = {g.fact for g in graded}
    rows = {
        row.fact: row
        for row in db.scalars(select(FactMemory).where(FactMemory.user_id == user.id, FactMemory.fact.in_(facts)))
    }
    for g in graded:
        row = rows.get(g.fact)
        if row is None:
            row = FactMemory(user_id=user.id, fact=g.fact, skill=g.skill, seen=0, correct=0, fluent=0)
            db.add(row)
            rows[g.fact] = row
        stats = record_attempt(_stats(row), correct=g.correct, fluent=g.fluent, latency_ms=g.latency_ms, now=now)
        row.skill = g.skill
        row.seen = stats.seen
        row.correct = stats.correct
        row.fluent = stats.fluent
        row.mean_latency_ms = stats.mean_latency_ms
        row.last_seen_at = stats.last_seen_at
