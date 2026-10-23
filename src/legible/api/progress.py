"""Loads and stores progression state, and builds the progress summaries.

Routers call :func:`record_session` once per completed session; it applies the pure
rules in :mod:`legible.progression.engine` and appends a :class:`TrackEvent`.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..progression import TRACKS, TrackSpec, TrackState, apply_session
from .models import TrackEvent, TrackStateRow, User
from .schemas import HistoryPoint, ProgressionUpdateOut, TrackOut, TrackParameterOut, TrackSnapshot, WeekSummary

HISTORY_DAYS = 365
WEEKS_SHOWN = 12


def spec_for(track_key: str) -> TrackSpec:
    try:
        return TRACKS[track_key]
    except KeyError:
        raise KeyError(f"unknown track {track_key!r}") from None


def _row_to_state(row: TrackStateRow) -> TrackState:
    return TrackState(
        level=row.level,
        streak=row.streak,
        best_streak=row.best_streak,
        sessions=row.sessions,
        perfect_sessions=row.perfect_sessions,
        struggle_count=row.struggle_count,
        last_promoted_on=row.last_promoted_on,
        promotions_on_last_date=row.promotions_on_last_date,
    )


def _state_row(db: Session, user_id: str, track_key: str, *, lock: bool = False) -> TrackStateRow | None:
    query = select(TrackStateRow).where(TrackStateRow.user_id == user_id, TrackStateRow.track_key == track_key)
    if lock:
        query = query.with_for_update()
    return db.scalars(query).first()


def get_level(db: Session, user_id: str, track_key: str) -> int:
    row = _state_row(db, user_id, track_key)
    return row.level if row else 0


def parameters_out(spec: TrackSpec, level: int) -> dict[str, TrackParameterOut]:
    return {name: TrackParameterOut(**info) for name, info in spec.describe_parameters(level).items()}


def snapshot(db: Session, user_id: str, track_key: str) -> TrackSnapshot:
    spec = spec_for(track_key)
    level = get_level(db, user_id, track_key)
    return TrackSnapshot(track=track_key, level=level, parameters=parameters_out(spec, level))


def record_session(db: Session, user: User, track_key: str, score: float, today: date) -> ProgressionUpdateOut:
    """Applies one session's score to a track and stages the changes on ``db`` (caller commits)."""
    spec = spec_for(track_key)
    row = _state_row(db, user.id, track_key, lock=True)
    if row is None:
        row = TrackStateRow(user_id=user.id, track_key=track_key)
        db.add(row)
        state = TrackState()
    else:
        state = _row_to_state(row)

    new_state, outcome = apply_session(spec, state, score, today)
    row.level = new_state.level
    row.streak = new_state.streak
    row.best_streak = new_state.best_streak
    row.sessions = new_state.sessions
    row.perfect_sessions = new_state.perfect_sessions
    row.struggle_count = new_state.struggle_count
    row.last_promoted_on = new_state.last_promoted_on
    row.promotions_on_last_date = new_state.promotions_on_last_date
    db.add(
        TrackEvent(
            user_id=user.id,
            track_key=track_key,
            local_date=today,
            score=outcome.score,
            perfect=outcome.perfect,
            level_before=outcome.previous_level,
            level_after=outcome.level,
            reason=outcome.reason,
        )
    )
    return ProgressionUpdateOut(
        track=track_key,
        previous_level=outcome.previous_level,
        level=outcome.level,
        promoted=outcome.promoted,
        demoted=outcome.demoted,
        reason=outcome.reason,
        message=outcome.message,
        progress=spec.progress(outcome.level),
        parameters=parameters_out(spec, outcome.level),
    )


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())  # Monday


def track_summaries(db: Session, user: User, today: date) -> list[TrackOut]:
    rows = {r.track_key: r for r in db.scalars(select(TrackStateRow).where(TrackStateRow.user_id == user.id))}
    since = today - timedelta(days=HISTORY_DAYS - 1)
    events_by_track: dict[str, list[TrackEvent]] = defaultdict(list)
    query = (
        select(TrackEvent)
        .where(TrackEvent.user_id == user.id, TrackEvent.local_date >= since)
        .order_by(TrackEvent.local_date, TrackEvent.id)
    )
    for event in db.scalars(query):
        events_by_track[event.track_key].append(event)

    out = []
    for key, spec in TRACKS.items():
        row = rows.get(key)
        state = _row_to_state(row) if row else TrackState()
        events = events_by_track.get(key, [])
        out.append(
            TrackOut(
                key=key,
                module=spec.module,
                label=spec.label,
                description=spec.description,
                level=state.level,
                max_level=spec.top_level,
                progress=spec.progress(state.level),
                streak=state.streak,
                best_streak=state.best_streak,
                sessions=state.sessions,
                perfect_sessions=state.perfect_sessions,
                last_promoted_on=state.last_promoted_on,
                weekly_goal=spec.weekly_goal,
                parameters=parameters_out(spec, state.level),
                history=_daily_history(events),
                weekly=_weekly(events, today),
            )
        )
    return out


def _daily_history(events: list[TrackEvent]) -> list[HistoryPoint]:
    """One point per practice day: the level at the end of the day and its best score."""
    by_day: dict[date, HistoryPoint] = {}
    for event in events:  # already in chronological order
        previous = by_day.get(event.local_date)
        best = max(previous.score, event.score) if previous else event.score
        by_day[event.local_date] = HistoryPoint(date=event.local_date, level=event.level_after, score=round(best, 4))
    return [by_day[d] for d in sorted(by_day)]


def _weekly(events: list[TrackEvent], today: date) -> list[WeekSummary]:
    """The last ``WEEKS_SHOWN`` weeks (Monday-based), oldest first, including empty weeks."""
    current = week_start(today)
    starts = [current - timedelta(weeks=i) for i in reversed(range(WEEKS_SHOWN))]
    buckets: dict[date, list[TrackEvent]] = {s: [] for s in starts}
    for event in events:
        start = week_start(event.local_date)
        if start in buckets:
            buckets[start].append(event)
    summaries = []
    for start in starts:
        items = buckets[start]
        summaries.append(
            WeekSummary(
                week_start=start,
                sessions=len(items),
                perfect_sessions=sum(e.perfect for e in items),
                levels_gained=sum(e.level_after - e.level_before for e in items),
                average_score=round(sum(e.score for e in items) / len(items), 4) if items else None,
            )
        )
    return summaries
