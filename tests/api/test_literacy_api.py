import io

import numpy as np
from PIL import Image

from legible.api.services.handwriting import ModelUnavailable

from ..conftest import register


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((40, 120), 255, np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_full_session_moves_both_tracks(client, auth_headers):
    session = client.get("/api/literacy/session", headers=auth_headers).json()
    assert session["skill"] == "short_a" and len(session["cards"]) == 9
    assert {t["track"] for t in session["tracks"]} == {"literacy.reading", "literacy.spelling"}
    for card in session["cards"]:
        response = client.post(f"/api/literacy/session/{session['session_id']}/answer", headers=auth_headers,
                               json={"card_id": card["card_id"], "response": card["word"], "latency_ms": 1500})
        assert response.status_code == 200 and response.json()["correct"] is True
    again = client.post(f"/api/literacy/session/{session['session_id']}/answer", headers=auth_headers,
                        json={"card_id": session["cards"][0]["card_id"], "response": "x", "latency_ms": 1})
    assert again.status_code == 409

    done = client.post(f"/api/literacy/session/{session['session_id']}/complete", headers=auth_headers).json()
    assert done["perfect"] and done["correct"] == done["total"] == 9
    assert {p["track"]: p["level"] for p in done["progressions"]} == {"literacy.reading": 1, "literacy.spelling": 1}
    twice = client.post(f"/api/literacy/session/{session['session_id']}/complete", headers=auth_headers)
    assert twice.status_code == 409


def test_mistakes_are_explained_and_hold_the_level(client, auth_headers):
    session = client.get("/api/literacy/session", headers=auth_headers).json()
    spell = next(c for c in session["cards"] if c["mode"] == "spell")
    result = client.post(f"/api/literacy/session/{session['session_id']}/answer", headers=auth_headers,
                         json={"card_id": spell["card_id"], "response": spell["word"][::-1], "latency_ms": 3000}).json()
    assert result["correct"] is False and result["expected"] == spell["word"] and result["feedback"]
    done = client.post(f"/api/literacy/session/{session['session_id']}/complete", headers=auth_headers).json()
    spelling = next(p for p in done["progressions"] if p["track"] == "literacy.spelling")
    assert spelling["level"] == 0 and not done["perfect"]


def test_sessions_are_private_and_validated(client, auth_headers):
    session = client.get("/api/literacy/session", headers=auth_headers).json()
    other = {"Authorization": f"Bearer {register(client)['access_token']}"}
    url = f"/api/literacy/session/{session['session_id']}"
    assert client.post(f"{url}/answer", headers=other, json={"card_id": "r0", "response": "x", "latency_ms": 1}).status_code == 404
    assert client.post(f"{url}/answer", headers=auth_headers, json={"card_id": "nope", "response": "x", "latency_ms": 1}).status_code == 404
    assert client.post(f"{url}/complete", headers=auth_headers).status_code == 409  # nothing answered yet


class FakeHandwriting:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    async def read_word(self, data, *, user_id=None):
        if self.error:
            raise self.error
        return self.text


def test_handwritten_answers(app, client, auth_headers):
    session = client.get("/api/literacy/session", headers=auth_headers).json()
    spell = [c for c in session["cards"] if c["mode"] == "spell"]
    url = f"/api/literacy/session/{session['session_id']}/answer-handwriting"

    app.state.handwriting = FakeHandwriting(text=spell[0]["word"].capitalize())
    ok = client.post(url, headers=auth_headers, files={"file": ("w.png", _png(), "image/png")},
                     data={"card_id": spell[0]["card_id"], "latency_ms": "2500"})
    assert ok.status_code == 200 and ok.json()["correct"] and ok.json()["recognized"] == spell[0]["word"].capitalize()

    app.state.handwriting = FakeHandwriting(error=ModelUnavailable("not set up"))
    down = client.post(url, headers=auth_headers, files={"file": ("w.png", _png(), "image/png")},
                       data={"card_id": spell[1]["card_id"]})
    assert down.status_code == 503
    wrong_type = client.post(url, headers=auth_headers, files={"file": ("w.gif", b"GIF89a", "image/gif")},
                             data={"card_id": spell[1]["card_id"]})
    assert wrong_type.status_code == 415
