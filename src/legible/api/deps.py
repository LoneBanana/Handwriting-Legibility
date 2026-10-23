"""FastAPI dependencies shared by every router."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .models import User
from .security import decode_access_token
from .settings import Settings

_bearer = HTTPBearer(auto_error=False)


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.sessionmaker()
    try:
        yield session
    finally:
        session.close()


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def get_current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None:
        raise _unauthorized("Please log in.")
    user_id = decode_access_token(credentials.credentials, request.app.state.secret_key)
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise _unauthorized("Your session has expired. Please log in again.")
    return user


def user_timezone(request: Request, x_timezone: str | None) -> ZoneInfo:
    for name in (x_timezone, request.app.state.settings.default_timezone, "UTC"):
        if not name:
            continue
        try:
            return ZoneInfo(name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return ZoneInfo("UTC")


def get_today(request: Request, x_timezone: Annotated[str | None, Header()] = None) -> date:
    """The learner's local calendar date (from the ``X-Timezone`` header the web app sends)."""
    return datetime.now(user_timezone(request, x_timezone)).date()


DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
Today = Annotated[date, Depends(get_today)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]
