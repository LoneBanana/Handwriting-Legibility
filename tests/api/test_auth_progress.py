from datetime import datetime, timezone

from legible.api.models import User
from legible.api.progress import record_session

from ..conftest import register


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_register_login_me(client):
    created = register(client, email="Sam@Example.com")
    assert created["user"]["email"] == "sam@example.com"
    assert created["user"]["settings"]["font"] == "default"

    response = client.post("/api/auth/login", json={"email": "sam@example.com", "password": "correct horse battery"})
    assert response.status_code == 200
    token = response.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["display_name"] == "Sam"


def test_register_rejects_short_password_and_duplicates(client):
    short = client.post("/api/auth/register", json={"email": "a@b.co", "password": "short", "display_name": "A"})
    assert short.status_code == 422
    register(client, email="dup@example.com")
    again = client.post(
        "/api/auth/register", json={"email": "DUP@example.com", "password": "long enough pw", "display_name": "B"}
    )
    assert again.status_code == 409


def test_bad_login_and_missing_token(client):
    register(client, email="x@example.com")
    assert client.post("/api/auth/login", json={"email": "x@example.com", "password": "nope nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "nope nope"}).status_code == 401
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_settings_patch_validates_and_merges(client, auth_headers):
    response = client.patch("/api/auth/me/settings", json={"font": "opendyslexic", "font_scale": 1.4}, headers=auth_headers)
    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["font"] == "opendyslexic" and settings["font_scale"] == 1.4 and settings["theme"] == "light"

    assert client.patch("/api/auth/me/settings", json={"font_scale": 9}, headers=auth_headers).status_code == 422
    assert client.patch("/api/auth/me/settings", json={"unknown": 1}, headers=auth_headers).status_code == 422


def test_delete_account(client, auth_headers):
    assert client.delete("/api/auth/me", headers=auth_headers).status_code == 204
    assert client.get("/api/auth/me", headers=auth_headers).status_code == 401


def test_progress_lists_every_track_with_weekly_meter(client, auth_headers):
    response = client.get("/api/progress", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    keys = {t["key"] for t in body["tracks"]}
    assert {"handwriting.legibility", "literacy.reading", "literacy.spelling", "numeracy.fluency",
            "converse.listening", "converse.expressions"} <= keys
    track = next(t for t in body["tracks"] if t["key"] == "converse.listening")
    assert track["level"] == 0 and track["parameters"]["words_per_minute"]["value"] == 110
    assert len(track["weekly"]) == 12 and track["weekly"][-1]["sessions"] == 0


def test_recorded_sessions_show_in_history_and_week(app, client, auth_headers):
    user_id = client.get("/api/auth/me", headers=auth_headers).json()["id"]
    today = datetime.now(timezone.utc).date()  # the fixture's X-Timezone is UTC
    with app.state.sessionmaker() as db:
        user = db.get(User, user_id)
        first = record_session(db, user, "numeracy.fluency", 1.0, today)
        second = record_session(db, user, "numeracy.fluency", 1.0, today)
        db.commit()
    assert first.promoted and first.level == 1
    assert second.reason == "daily_cap" and second.level == 1

    track = next(t for t in client.get("/api/progress", headers=auth_headers).json()["tracks"]
                 if t["key"] == "numeracy.fluency")
    assert track["level"] == 1 and track["streak"] == 2 and track["sessions"] == 2
    assert track["history"] == [{"date": today.isoformat(), "level": 1, "score": 1.0}]
    assert track["weekly"][-1]["sessions"] == 2 and track["weekly"][-1]["levels_gained"] == 1


def test_login_attempts_are_rate_limited(client):
    register(client, email="limit@example.com")
    for _ in range(10):
        client.post("/api/auth/login", json={"email": "limit@example.com", "password": "wrong password"})
    blocked = client.post("/api/auth/login", json={"email": "limit@example.com", "password": "correct horse battery"})
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers


def test_validation_errors_never_crash_on_non_finite_numbers(client, auth_headers):
    body = '{"font_scale": Infinity}'
    response = client.patch("/api/auth/me/settings", content=body,
                            headers={**auth_headers, "Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["detail"][0]["loc"]
