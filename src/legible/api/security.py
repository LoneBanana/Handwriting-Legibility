"""Password hashing (Argon2id) and bearer tokens (JWT, HS256)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()
_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def create_access_token(user_id: str, secret: str, ttl_hours: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "iat": now, "exp": now + timedelta(hours=ttl_hours), "typ": "access"}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_access_token(token: str, secret: str) -> str | None:
    """Returns the user id, or None for a missing, expired, or tampered token."""
    try:
        payload = jwt.decode(token, secret, algorithms=[_ALGORITHM], options={"require": ["sub", "exp"]})
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "access" or not isinstance(payload.get("sub"), str):
        return None
    return payload["sub"]
