"""Request and response models shared across routers (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FontChoice = Literal["default", "lexend", "atkinson", "opendyslexic"]
ThemeChoice = Literal["light", "dark", "cream", "high-contrast"]


class UserSettings(BaseModel):
    font: FontChoice = "default"
    font_scale: float = Field(1.0, ge=0.8, le=2.0)
    line_spacing: float = Field(1.6, ge=1.2, le=2.5)
    theme: ThemeChoice = "light"
    reduce_motion: bool = False
    tts_voice: str | None = Field(None, max_length=200)
    store_handwriting_samples: bool = False


class UserSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    font: FontChoice | None = None
    font_scale: float | None = Field(None, ge=0.8, le=2.0)
    line_spacing: float | None = Field(None, ge=1.2, le=2.5)
    theme: ThemeChoice | None = None
    reduce_motion: bool | None = None
    tts_voice: str | None = Field(None, max_length=200)
    store_handwriting_samples: bool | None = None


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: datetime
    settings: UserSettings


class RegisterIn(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    display_name: str = Field(min_length=1, max_length=80)


class LoginIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=256)


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserOut


# ------------------------------------------------------------------------ progression


class TrackParameterOut(BaseModel):
    value: float
    start: float
    target: float
    unit: str
    label: str


class ProgressionUpdateOut(BaseModel):
    track: str
    previous_level: int
    level: int
    promoted: bool
    demoted: bool
    reason: Literal["perfect", "held", "daily_cap", "struggling", "max_level"]
    message: str
    progress: float
    parameters: dict[str, TrackParameterOut]


class TrackSnapshot(BaseModel):
    """A track's level and current parameter values, sent when a session starts."""

    track: str
    level: int
    parameters: dict[str, TrackParameterOut]


class HistoryPoint(BaseModel):
    date: date
    level: int
    score: float


class WeekSummary(BaseModel):
    week_start: date
    sessions: int
    perfect_sessions: int
    levels_gained: int
    average_score: float | None


class TrackOut(BaseModel):
    key: str
    module: Literal["handwriting", "literacy", "numeracy", "converse"]
    label: str
    description: str
    level: int
    max_level: int
    progress: float
    streak: int
    best_streak: int
    sessions: int
    perfect_sessions: int
    last_promoted_on: date | None
    weekly_goal: int
    parameters: dict[str, TrackParameterOut]
    history: list[HistoryPoint]
    weekly: list[WeekSummary]


class ProgressOut(BaseModel):
    week_start: date
    tracks: list[TrackOut]
