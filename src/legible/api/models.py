"""Database tables for every module.

Module-specific session content (flash cards, math problems, comprehension questions)
is stored as JSON on the row that owns it: it is always read and written as a whole,
and keeping answers server-side means the client never sees them before it answers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, new_id, utcnow


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(80))
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ------------------------------------------------------------------ progression (Phase 5)


class TrackStateRow(Base):
    __tablename__ = "track_states"
    __table_args__ = (UniqueConstraint("user_id", "track_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    track_key: Mapped[str] = mapped_column(String(64))
    level: Mapped[int] = mapped_column(Integer, default=0)
    streak: Mapped[int] = mapped_column(Integer, default=0)
    best_streak: Mapped[int] = mapped_column(Integer, default=0)
    sessions: Mapped[int] = mapped_column(Integer, default=0)
    perfect_sessions: Mapped[int] = mapped_column(Integer, default=0)
    struggle_count: Mapped[int] = mapped_column(Integer, default=0)
    last_promoted_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    promotions_on_last_date: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class TrackEvent(Base):
    """One completed session on one track: the source of history charts and weekly meters."""

    __tablename__ = "track_events"
    __table_args__ = (Index("ix_track_events_user_track_date", "user_id", "track_key", "local_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    track_key: Mapped[str] = mapped_column(String(64))
    local_date: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    score: Mapped[float] = mapped_column(Float)
    perfect: Mapped[bool] = mapped_column(Boolean)
    level_before: Mapped[int] = mapped_column(Integer)
    level_after: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(16))


# ------------------------------------------------------------------ handwriting (Phase 1)


class HandwritingSample(Base):
    """A transcribed photo kept (with consent) so corrections can personalize the model."""

    __tablename__ = "handwriting_samples"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    image_path: Mapped[str] = mapped_column(String(512))
    lines: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # [{bbox, text, confidence}]
    text: Mapped[str] = mapped_column(Text, default="")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    used_in_adapter: Mapped[bool] = mapped_column(Boolean, default=False)


class PersonalAdapter(Base):
    """A per-user low-rank adapter fine-tuned on that user's corrected handwriting."""

    __tablename__ = "personal_adapters"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    path: Mapped[str] = mapped_column(String(512))
    base_model_version: Mapped[str] = mapped_column(String(64))
    samples: Mapped[int] = mapped_column(Integer)
    lines: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ------------------------------------------------------------- literacy & numeracy (2, 3)


class PracticeSession(Base):
    """A literacy card session or a numeracy round, with its answer key and responses."""

    __tablename__ = "practice_sessions"
    __table_args__ = (Index("ix_practice_sessions_user_kind", "user_id", "kind"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))  # "literacy" | "numeracy"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    responses: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class WordMemory(Base):
    """Spaced-repetition state for one word in one mode (read or spell) for one learner."""

    __tablename__ = "word_memory"
    __table_args__ = (UniqueConstraint("user_id", "word", "mode"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    word: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(8))
    skill: Mapped[str] = mapped_column(String(64))
    seen: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    wrong: Mapped[int] = mapped_column(Integer, default=0)
    half_life_days: Mapped[float] = mapped_column(Float, default=1.0)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class FactMemory(Base):
    """Accuracy and speed history for one math fact (e.g. "mul:7x8") for one learner."""

    __tablename__ = "fact_memory"
    __table_args__ = (UniqueConstraint("user_id", "fact"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    fact: Mapped[str] = mapped_column(String(64))
    skill: Mapped[str] = mapped_column(String(64))
    seen: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    fluent: Mapped[int] = mapped_column(Integer, default=0)
    mean_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ------------------------------------------------------------------- conversation (Phase 4)


class ConverseSessionRow(Base):
    __tablename__ = "converse_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(16))  # "local" | "tavus"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # pace, intensity, captions
    tavus_conversation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ConverseTurn(Base):
    __tablename__ = "converse_turns"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("converse_sessions.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    user_text: Mapped[str] = mapped_column(Text)
    assistant_text: Mapped[str] = mapped_column(Text, default="")
    segments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # [{index, emotion, text}]
    complete: Mapped[bool] = mapped_column(Boolean, default=False)


class ComprehensionCheckRow(Base):
    __tablename__ = "comprehension_checks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    turn_id: Mapped[str] = mapped_column(ForeignKey("converse_turns.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    questions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # includes answers
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
