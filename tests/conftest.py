"""Shared fixtures: an isolated app (temp data dir, in-memory SQLite) and logged-in users."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from legible.api import ratelimit
from legible.api.main import create_app
from legible.api.settings import Settings

_emails = itertools.count()


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    for limiter in (ratelimit.login_limiter, ratelimit.register_limiter, ratelimit.llm_limiter):
        limiter.reset()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        data_dir=tmp_path / "data",
        database_url="sqlite://",
        jwt_secret="test-secret-test-secret-test-secret",
        llm_mode="off",
        serve_frontend=False,
        default_timezone="UTC",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, email: str | None = None, password: str = "correct horse battery") -> dict:
    email = email or f"learner{next(_emails)}@example.com"
    response = client.post(
        "/api/auth/register", json={"email": email, "password": password, "display_name": "Sam"}
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    token = register(client)["access_token"]
    return {"Authorization": f"Bearer {token}", "X-Timezone": "UTC"}
