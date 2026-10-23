"""Phase 5: the learner's tracks, weekly meters, and yearly progress."""

from __future__ import annotations

from fastapi import APIRouter

from ..deps import CurrentUser, DbSession, Today
from ..progress import track_summaries, week_start
from ..schemas import ProgressOut

router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("", response_model=ProgressOut)
def get_progress(user: CurrentUser, db: DbSession, today: Today) -> ProgressOut:
    return ProgressOut(week_start=week_start(today), tracks=track_summaries(db, user, today))
