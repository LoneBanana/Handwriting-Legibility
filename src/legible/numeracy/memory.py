"""What Legible remembers about each fact for each learner, and how that steers practice.

Per learner and fact the API stores (in ``FactMemory``) how often it was seen, answered
correctly, and answered correctly *within the time limit* ("fluent"), plus a recent
average of how long correct answers took and when it was last seen.

:func:`priority` turns that into "how much does this fact need practice now":

1. facts that are often missed or slow come first (they need the most work);
2. then facts never seen, and facts not seen for a couple of weeks (spaced practice:
   a gap before review makes memories last longer);
3. facts answered quickly and correctly not long ago come last.

A little randomness keeps rounds varied among facts that need practice equally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone

MISS_WEIGHT = 3.0          # need points for a fact that is always answered wrongly
SLOW_WEIGHT = 2.0          # ... for one that is never answered within the time limit
SLOW_AVERAGE_WEIGHT = 1.0  # ... extra when its usual correct answer takes longer than the limit
UNSEEN_PRIORITY = 1.0      # a fact never seen is worth meeting, but less than a weak one
SPACING_DAYS = 14          # after this long unseen, a known fact is as due as a new one
JITTER = 0.5               # random spread among facts with similar priority
LATENCY_WEIGHT = 0.3       # how much the newest correct answer moves the average time
LATENCY_CAP_MS = 60_000    # a learner who stepped away should not skew the average forever


@dataclass(frozen=True)
class FactStats:
    seen: int = 0
    correct: int = 0
    fluent: int = 0                     # correct and within the time limit
    mean_latency_ms: float | None = None  # recent average time of correct answers
    last_seen_at: datetime | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.seen if self.seen else 0.0

    @property
    def fluency(self) -> float:
        return self.fluent / self.seen if self.seen else 0.0


def need(stats: FactStats, time_limit_ms: float) -> float:
    """How much a fact that has been seen needs work: 0 when always quick and correct, up to 6."""
    slow_on_average = stats.mean_latency_ms is not None and stats.mean_latency_ms > time_limit_ms
    return (
        MISS_WEIGHT * (1 - stats.accuracy)
        + SLOW_WEIGHT * (1 - stats.fluency)
        + (SLOW_AVERAGE_WEIGHT if slow_on_average else 0.0)
    )


def days_since(moment: datetime | None, now: datetime) -> float | None:
    if moment is None:
        return None
    if moment.tzinfo is None:  # SQLite hands back naive datetimes; they are stored in UTC
        moment = moment.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max((now - moment).total_seconds() / 86_400, 0.0)


def priority(
    stats: FactStats | None,
    *,
    now: datetime,
    time_limit_ms: float,
    rng: random.Random,
) -> float:
    """Higher means practise sooner.

    New facts score 1.0 to 1.5, so a fact whose :func:`need` is above 1.5 (say, missed
    half the time) always comes before them, however the random spread falls.
    """
    if stats is None or stats.seen == 0:
        base = UNSEEN_PRIORITY
    else:
        days = days_since(stats.last_seen_at, now)
        due = 1.0 if days is None else min(days / SPACING_DAYS, 1.0)
        base = need(stats, time_limit_ms) + due
    return base + rng.uniform(0, JITTER)


def record_attempt(
    stats: FactStats,
    *,
    correct: bool,
    fluent: bool,
    latency_ms: float | None,
    now: datetime,
) -> FactStats:
    """Folds one answer into a fact's history.

    The average time only follows correct answers ("how long it takes you to get it
    right") and is exponentially weighted, so getting faster shows up within a few rounds.
    """
    mean = stats.mean_latency_ms
    if correct and latency_ms is not None:
        latency = min(max(latency_ms, 0.0), LATENCY_CAP_MS)
        mean = latency if mean is None else mean + LATENCY_WEIGHT * (latency - mean)
    return replace(
        stats,
        seen=stats.seen + 1,
        correct=stats.correct + int(correct),
        fluent=stats.fluent + int(fluent),
        mean_latency_ms=mean,
        last_seen_at=now,
    )
