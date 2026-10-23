"""Half-life memory model for spaced review of words.

In the spirit of half-life regression (Settles & Meeder, 2016): the chance a learner
still knows a word decays as ``p = 2 ** (-days_since_seen / half_life)``. A correct
answer stretches the half-life (more for a quick answer, since fluent recall is
stronger memory) and a mistake shrinks it, so words that slip come back sooner and
mastered words space out. Sessions review the words with the lowest ``p`` first.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

MIN_HALF_LIFE = 0.25    # days
MAX_HALF_LIFE = 180.0
INITIAL_HALF_LIFE = 1.0
GROWTH = 2.2            # correct answer
FLUENT_BONUS = 1.3      # extra growth for a quick correct answer
SHRINK = 0.4            # mistake
FLUENT_MS = 4000        # a "quick" answer for spelling/reading cards


@dataclass(frozen=True)
class MemoryState:
    seen: int = 0
    correct: int = 0
    wrong: int = 0
    half_life_days: float = INITIAL_HALF_LIFE
    last_seen_at: datetime | None = None


def recall_probability(state: MemoryState, now: datetime) -> float:
    """Predicted chance of recalling the word now; 0 for never-seen words."""
    if state.last_seen_at is None or state.seen == 0:
        return 0.0
    days = max((now - state.last_seen_at).total_seconds() / 86400.0, 0.0)
    return 2.0 ** (-days / max(state.half_life_days, MIN_HALF_LIFE))


def update(state: MemoryState, correct: bool, latency_ms: float, now: datetime) -> MemoryState:
    if correct:
        factor = GROWTH * (FLUENT_BONUS if latency_ms <= FLUENT_MS else 1.0)
        half_life = state.half_life_days * factor
    else:
        half_life = state.half_life_days * SHRINK
    return replace(
        state,
        seen=state.seen + 1,
        correct=state.correct + int(correct),
        wrong=state.wrong + int(not correct),
        half_life_days=min(max(half_life, MIN_HALF_LIFE), MAX_HALF_LIFE),
        last_seen_at=now,
    )
