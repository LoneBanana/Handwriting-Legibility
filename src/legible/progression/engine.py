"""Mastery-gated progression shared by every module (Phase 5).

Rules, applied once per completed session:

* a **perfect** session (score >= ``promote_threshold``) promotes the track one level,
  at most ``max_promotions_per_day`` times per calendar day in the learner's time zone;
* any imperfect session **holds** the level, so difficulty never rises on a shaky day;
* ``struggle_window`` consecutive sessions below ``struggle_threshold`` step the level
  **back** by one, so a learner is never left practising at a level that frustrates.

Everything here is pure (no I/O); the API layer loads and stores :class:`TrackState`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal, Mapping

from .curves import Curve

Reason = Literal["perfect", "held", "daily_cap", "struggling", "max_level"]


@dataclass(frozen=True)
class TrackSpec:
    key: str
    module: Literal["handwriting", "literacy", "numeracy", "converse"]
    label: str
    description: str
    parameters: Mapping[str, Curve] = field(default_factory=dict)
    levels_to_target: int = 300          # about a year of near-daily perfect sessions
    max_level: int | None = None         # defaults to levels_to_target
    promote_threshold: float = 1.0
    struggle_threshold: float = 0.6
    struggle_window: int = 3
    max_promotions_per_day: int = 1
    weekly_goal: int = 5                 # sessions per week shown on the weekly meter
    promotion_note: str = ""             # what changes next time, e.g. "Words flash a little faster."

    @property
    def top_level(self) -> int:
        return self.levels_to_target if self.max_level is None else self.max_level

    def parameter_values(self, level: int) -> dict[str, float]:
        return {name: curve.at(level, self.levels_to_target) for name, curve in self.parameters.items()}

    def describe_parameters(self, level: int) -> dict[str, dict[str, float | str]]:
        return {
            name: {
                "value": curve.at(level, self.levels_to_target),
                "start": curve.start,
                "target": curve.target,
                "unit": curve.unit,
                "label": curve.label,
            }
            for name, curve in self.parameters.items()
        }

    def progress(self, level: int) -> float:
        """Fraction of the way to this track's yearly target, in [0, 1]."""
        if self.levels_to_target <= 0:
            return 1.0
        return min(max(level, 0) / self.levels_to_target, 1.0)


@dataclass(frozen=True)
class TrackState:
    level: int = 0
    streak: int = 0                      # consecutive perfect sessions
    best_streak: int = 0
    sessions: int = 0
    perfect_sessions: int = 0
    struggle_count: int = 0              # consecutive sessions below the struggle threshold
    last_promoted_on: date | None = None
    promotions_on_last_date: int = 0


@dataclass(frozen=True)
class Outcome:
    track: str
    score: float
    perfect: bool
    previous_level: int
    level: int
    reason: Reason
    message: str

    @property
    def promoted(self) -> bool:
        return self.level > self.previous_level

    @property
    def demoted(self) -> bool:
        return self.level < self.previous_level


def clamp_score(score: float) -> float:
    if score is None or not math.isfinite(score):
        return 0.0
    return min(max(float(score), 0.0), 1.0)


def apply_session(spec: TrackSpec, state: TrackState, score: float, today: date) -> tuple[TrackState, Outcome]:
    """Fold one completed session's ``score`` (0..1) into ``state``."""
    score = clamp_score(score)
    perfect = score >= spec.promote_threshold - 1e-9
    level = state.level
    sessions = state.sessions + 1

    if perfect:
        streak = state.streak + 1
        promotions_today = state.promotions_on_last_date if state.last_promoted_on == today else 0
        new = replace(
            state,
            sessions=sessions,
            perfect_sessions=state.perfect_sessions + 1,
            streak=streak,
            best_streak=max(state.best_streak, streak),
            struggle_count=0,
        )
        if level >= spec.top_level:
            reason: Reason = "max_level"
        elif promotions_today >= spec.max_promotions_per_day:
            reason = "daily_cap"
        else:
            reason = "perfect"
            new = replace(new, level=level + 1, last_promoted_on=today, promotions_on_last_date=promotions_today + 1)
    else:
        struggling = score < spec.struggle_threshold
        struggle_count = state.struggle_count + 1 if struggling else 0
        reason = "held"
        new_level = level
        if struggling and struggle_count >= spec.struggle_window:
            struggle_count = 0
            if level > 0:
                new_level = level - 1
                reason = "struggling"
        new = replace(state, sessions=sessions, streak=0, struggle_count=struggle_count, level=new_level)

    outcome = Outcome(
        track=spec.key,
        score=score,
        perfect=perfect,
        previous_level=level,
        level=new.level,
        reason=reason,
        message=_message(spec, reason, score, new.level),
    )
    return new, outcome


def _message(spec: TrackSpec, reason: Reason, score: float, level: int) -> str:
    if reason == "perfect":
        note = f" {spec.promotion_note}" if spec.promotion_note else ""
        return f"Perfect session — welcome to level {level}!{note}"
    if reason == "daily_cap":
        return "Perfect again! You already levelled up today, so the next step unlocks tomorrow."
    if reason == "max_level":
        return "Perfect — you have reached the top level for this skill."
    if reason == "struggling":
        return "We made things a little easier, so practice feels comfortable again."
    if score >= 0.8:
        return "So close! One perfect session moves you up a level."
    return "Good practice. Every session builds the skill, and the level waits until you are ready."
