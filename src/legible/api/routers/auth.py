"""Accounts: register, log in, profile settings, and deleting an account with its data."""

from __future__ import annotations

import logging
import re
import shutil

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import ValidationError
from sqlalchemy import select

from ..deps import AppSettings, CurrentUser, DbSession
from ..models import User
from ..ratelimit import login_limiter, register_limiter
from ..schemas import AuthResponse, LoginIn, RegisterIn, UserOut, UserSettings, UserSettingsPatch
from ..security import create_access_token, hash_password, needs_rehash, verify_password

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Verifying against a real hash when the email is unknown keeps response times similar,
# so login timing does not reveal which emails have accounts.
_DUMMY_HASH = hash_password("not-a-real-password")


def load_settings(user: User) -> UserSettings:
    try:
        return UserSettings.model_validate(user.settings or {})
    except ValidationError:
        log.warning("stored settings for user %s were invalid; using defaults", user.id)
        return UserSettings()


def user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        settings=load_settings(user),
    )


def _auth_response(request: Request, user: User) -> AuthResponse:
    settings = request.app.state.settings
    token = create_access_token(user.id, request.app.state.secret_key, settings.jwt_ttl_hours)
    return AuthResponse(access_token=token, user=user_out(user))


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, request: Request, db: DbSession) -> AuthResponse:
    register_limiter.hit(_client(request))
    email = _normalize_email(body.email)
    if not _EMAIL.match(email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Please enter a valid email address.")
    display_name = body.display_name.strip()
    if not display_name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Please tell us what to call you.")
    if db.scalars(select(User).where(User.email == email)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, detail="An account with that email already exists. Try logging in.")
    user = User(
        email=email,
        password_hash=hash_password(body.password),
        display_name=display_name,
        settings=UserSettings().model_dump(),
    )
    db.add(user)
    db.commit()
    return _auth_response(request, user)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginIn, request: Request, db: DbSession) -> AuthResponse:
    login_limiter.hit(f"{_normalize_email(body.email)}|{_client(request)}",
                      "Too many login attempts. Please wait a few minutes and try again.")
    user = db.scalars(select(User).where(User.email == _normalize_email(body.email))).first()
    if user is None:
        verify_password(_DUMMY_HASH, body.password)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="That email and password don’t match.")
    if not verify_password(user.password_hash, body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="That email and password don’t match.")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        db.commit()
    return _auth_response(request, user)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return user_out(user)


@router.patch("/me/settings", response_model=UserOut)
def update_settings(patch: UserSettingsPatch, user: CurrentUser, db: DbSession) -> UserOut:
    merged = {**load_settings(user).model_dump(), **patch.model_dump(exclude_unset=True)}
    user.settings = UserSettings.model_validate(merged).model_dump()
    db.commit()
    return user_out(user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(user: CurrentUser, db: DbSession, settings: AppSettings) -> Response:
    """Deletes the account, all progress, and every stored handwriting sample and adapter."""
    user_id = user.id
    db.delete(user)
    db.commit()
    for directory in (settings.uploads_dir / user_id, settings.adapters_dir / user_id):
        shutil.rmtree(directory, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
