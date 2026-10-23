"""Conversation API: sessions (local and Tavus), the SSE answer stream, checks, and progress."""

import json

import httpx
import pytest
from sqlalchemy import select

from legible.api import ratelimit
from legible.api.models import ComprehensionCheckRow, ConverseSessionRow, ConverseTurn, TrackStateRow
from legible.api.services.llm import LLMError, UnavailableLLM
from legible.converse.prompts import GUIDE_SYSTEM_PROMPT
from legible.converse.tavus import TavusClient

from ..conftest import register
from ..converse.fakes import FakeLLM, chunked

ANSWER = (
    "[excited] Great question! [surprised] An octopus has three hearts. "
    "[content] Two of them pump blood through the gills. [neutral] The third pumps it around the body."
)
SEGMENTS = [
    ("excited", "Great question!"),
    ("surprised", "An octopus has three hearts."),
    ("content", "Two of them pump blood through the gills."),
    ("neutral", "The third pumps it around the body."),
]
CHECK = {
    "questions": [
        {
            "kind": "main_idea",
            "question": "What was the answer mostly about?",
            "correct_answer": "An octopus's hearts",
            "wrong_answers": ["Shark teeth", "Whale songs"],
            "explanation": "The guide talked about an octopus's three hearts.",
        },
        {
            "kind": "detail",
            "question": "How many hearts does an octopus have?",
            "correct_answer": "Three",
            "wrong_answers": ["One", "Two"],
            "explanation": "The guide said an octopus has three hearts.",
        },
    ]
}


@pytest.fixture(autouse=True)
def _no_real_tavus(app):
    """Tavus is only ever a MockTransport here, even if LEGIBLE_TAVUS_* is set where tests run."""
    app.state.settings = app.state.settings.model_copy(
        update={"tavus_api_key": None, "tavus_replica_id": None, "tavus_persona_id": None}
    )


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


def start(client, headers, **body) -> dict:
    response = client.post("/api/converse/sessions", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def ask(client, headers, session_id: str, text: str = "How many hearts does an octopus have?") -> list[tuple[str, dict]]:
    response = client.post(f"/api/converse/sessions/{session_id}/messages", json={"text": text}, headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def check(client, headers, session_id: str, turn_id: str):
    return client.post(f"/api/converse/sessions/{session_id}/turns/{turn_id}/check", headers=headers)


def submit(client, headers, check_id: str, answers: dict[str, str]):
    body = {"answers": [{"question_id": qid, "choice": choice} for qid, choice in answers.items()]}
    return client.post(f"/api/converse/checks/{check_id}/submit", json=body, headers=headers)


def answer_key(app, check_id: str) -> dict[str, str]:
    with app.state.sessionmaker() as db:
        row = db.get(ComprehensionCheckRow, check_id)
        return {question["question_id"]: question["answer"] for question in row.questions}


def other_user(client) -> dict[str, str]:
    return {"Authorization": f"Bearer {register(client)['access_token']}", "X-Timezone": "UTC"}


def user_id(client, headers) -> str:
    return client.get("/api/auth/me", headers=headers).json()["id"]


# ------------------------------------------------------------------------------ sessions


def test_new_session_starts_slow_captioned_and_exaggerated(client, auth_headers):
    session = start(client, auth_headers)
    assert session["provider"] == "local" and session["tavus"] is None
    assert session["assistant"] == "demo"  # the test app has no Claude credentials
    assert session["words_per_minute"] == 110.0
    assert session["expression_intensity"] == 1.0
    assert session["captions"] == "full"
    assert [track["track"] for track in session["tracks"]] == ["converse.listening", "converse.expressions"]
    assert all(track["level"] == 0 for track in session["tracks"])
    assert client.post("/api/converse/sessions", json={}).status_code == 401


def test_session_settings_follow_the_learners_levels(app, client, auth_headers):
    uid = user_id(client, auth_headers)
    with app.state.sessionmaker() as db:
        db.add(TrackStateRow(user_id=uid, track_key="converse.listening", level=150))
        db.add(TrackStateRow(user_id=uid, track_key="converse.expressions", level=300))
        db.commit()
    session = start(client, auth_headers)
    assert session["words_per_minute"] == round(110 * (180 / 110) ** 0.5, 1)
    assert session["captions"] == "delayed"
    assert session["expression_intensity"] == 0.3
    with app.state.sessionmaker() as db:
        stored = db.get(ConverseSessionRow, session["session_id"]).settings
    assert stored["questions"] == 3 and stored["emotion_choices"] == 6


def test_assistant_is_claude_when_claude_is_configured(app, client, auth_headers):
    app.state.llm = FakeLLM()
    assert start(client, auth_headers)["assistant"] == "claude"


def test_tavus_requested_but_not_configured_falls_back_to_local(client, auth_headers):
    session = start(client, auth_headers, provider="tavus")
    assert session["provider"] == "local" and session["tavus"] is None
    assert client.post("/api/converse/sessions", json={"provider": "zoom"}, headers=auth_headers).status_code == 422


def test_tavus_session_is_created_and_ended(app, client, auth_headers):
    calls: list[httpx.Request] = []

    def tavus_api(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v2/conversations":
            return httpx.Response(200, json={
                "conversation_id": "c42", "conversation_url": "https://tavus.daily.co/c42",
                "status": "active", "meeting_token": "token-42",
            })
        return httpx.Response(200)

    app.state.tavus = TavusClient("key", face_id="r1", pal_id="p1", transport=httpx.MockTransport(tavus_api))
    session = start(client, auth_headers, provider="tavus")
    assert session["provider"] == "tavus"
    assert session["tavus"] == {
        "conversation_id": "c42", "conversation_url": "https://tavus.daily.co/c42", "meeting_token": "token-42",
    }
    with app.state.sessionmaker() as db:
        assert db.get(ConverseSessionRow, session["session_id"]).tavus_conversation_id == "c42"

    sid = session["session_id"]
    assert client.post(f"/api/converse/sessions/{sid}/end", headers=auth_headers).json() == {"ok": True}
    assert client.post(f"/api/converse/sessions/{sid}/end", headers=auth_headers).json() == {"ok": True}
    assert [(call.method, call.url.path) for call in calls] == [
        ("POST", "/v2/conversations"),
        ("POST", "/v2/conversations/c42/end"),  # ending twice tells Tavus once
    ]


def test_tavus_failure_falls_back_to_local(app, client, auth_headers):
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "Internal error"})

    app.state.tavus = TavusClient("key", face_id="r1", pal_id="p1", transport=httpx.MockTransport(broken))
    session = start(client, auth_headers, provider="tavus")
    assert session["provider"] == "local" and session["tavus"] is None
    with app.state.sessionmaker() as db:
        assert db.get(ConverseSessionRow, session["session_id"]).provider == "local"


def test_end_session_is_owned_and_idempotent(app, client, auth_headers):
    sid = start(client, auth_headers)["session_id"]
    assert client.post(f"/api/converse/sessions/{sid}/end", headers=other_user(client)).status_code == 404
    assert client.post("/api/converse/sessions/nope/end", headers=auth_headers).status_code == 404
    assert client.post(f"/api/converse/sessions/{sid}/end", headers=auth_headers).status_code == 200
    with app.state.sessionmaker() as db:
        assert db.get(ConverseSessionRow, sid).ended_at is not None


# ------------------------------------------------------------------------------ messages


def test_answer_streams_one_segment_per_sentence_then_done(app, client, auth_headers):
    llm = FakeLLM(chunks=chunked(ANSWER, 7))  # tags split across chunks
    app.state.llm = llm
    sid = start(client, auth_headers)["session_id"]
    events = ask(client, auth_headers, sid)

    assert [name for name, _ in events] == ["segment"] * 4 + ["done"]
    turn_id = events[0][1]["turn_id"]
    assert [(data["index"], data["emotion"], data["text"]) for _, data in events[:4]] == [
        (i, emotion, text) for i, (emotion, text) in enumerate(SEGMENTS)
    ]
    done = events[-1][1]
    assert done["turn_id"] == turn_id and all(data["turn_id"] == turn_id for _, data in events)
    assert done["text"] == " ".join(text for _, text in SEGMENTS)
    assert done["echo_text"] == " ".join(f'<emotion value="{emotion}"/> {text}' for emotion, text in SEGMENTS)

    call = llm.stream_calls[0]
    assert call["system"] == GUIDE_SYSTEM_PROMPT and call["effort"] == "low"
    assert call["messages"] == [{"role": "user", "content": "How many hearts does an octopus have?"}]
    with app.state.sessionmaker() as db:
        turn = db.get(ConverseTurn, turn_id)
        assert turn.complete and turn.assistant_text == done["text"]
        assert turn.segments[1] == {"index": 1, "emotion": "surprised", "text": "An octopus has three hearts."}


def test_follow_ups_include_earlier_turns_as_context(app, client, auth_headers):
    llm = FakeLLM(chunks=[ANSWER])
    app.state.llm = llm
    sid = start(client, auth_headers)["session_id"]
    ask(client, auth_headers, sid, "How many hearts does an octopus have?")
    ask(client, auth_headers, sid, "Why three?")
    assert llm.stream_calls[1]["messages"] == [
        {"role": "user", "content": "How many hearts does an octopus have?"},
        {"role": "assistant", "content": " ".join(f"[{emotion}] {text}" for emotion, text in SEGMENTS)},
        {"role": "user", "content": "Why three?"},
    ]


def test_demo_mode_answers_offline(client, auth_headers):
    sid = start(client, auth_headers)["session_id"]
    events = ask(client, auth_headers, sid, "Why is the sky blue?")
    assert events[-1][0] == "done" and "sky glows blue" in events[-1][1]["text"]
    assert {data["emotion"] for name, data in events if name == "segment"} >= {"content", "surprised"}

    events = ask(client, auth_headers, sid, "What is the capital of France?")
    assert "demo mode" in events[-1][1]["text"]


def test_claude_failure_mid_answer_sends_an_error_event(app, client, auth_headers):
    app.state.llm = FakeLLM(chunks=["[excited] Great question! [content] Octo", LLMError("Claude is busy right now.")])
    sid = start(client, auth_headers)["session_id"]
    events = ask(client, auth_headers, sid)
    assert events == [
        ("segment", {"turn_id": events[0][1]["turn_id"], "index": 0, "emotion": "excited", "text": "Great question!"}),
        ("error", {"detail": "Claude is busy right now."}),
    ]
    turn_id = events[0][1]["turn_id"]
    with app.state.sessionmaker() as db:
        assert db.get(ConverseTurn, turn_id).complete is False
    assert check(client, auth_headers, sid, turn_id).status_code == 409  # nothing finished to ask about


def test_an_empty_answer_is_an_error(app, client, auth_headers):
    app.state.llm = FakeLLM(chunks=["  ", "[content]"])
    sid = start(client, auth_headers)["session_id"]
    events = ask(client, auth_headers, sid)
    assert [name for name, _ in events] == ["error"]


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {"text": "x" * 501}, {}])
def test_message_text_is_validated(client, auth_headers, body):
    sid = start(client, auth_headers)["session_id"]
    assert client.post(f"/api/converse/sessions/{sid}/messages", json=body, headers=auth_headers).status_code == 422


def test_message_needs_an_open_session_of_ones_own(client, auth_headers):
    sid = start(client, auth_headers)["session_id"]
    body = {"text": "Hello"}
    assert client.post(f"/api/converse/sessions/{sid}/messages", json=body, headers=other_user(client)).status_code == 404
    assert client.post("/api/converse/sessions/missing/messages", json=body, headers=auth_headers).status_code == 404
    assert client.post(f"/api/converse/sessions/{sid}/messages", json=body).status_code == 401
    client.post(f"/api/converse/sessions/{sid}/end", headers=auth_headers)
    assert client.post(f"/api/converse/sessions/{sid}/messages", json=body, headers=auth_headers).status_code == 409


def test_claude_answers_are_rate_limited_but_the_demo_is_not(app, client, auth_headers, monkeypatch):
    monkeypatch.setattr(ratelimit.llm_limiter, "limit", 1)
    llm = FakeLLM(chunks=[ANSWER])
    app.state.llm = llm
    sid = start(client, auth_headers)["session_id"]
    ask(client, auth_headers, sid)
    limited = client.post(f"/api/converse/sessions/{sid}/messages", json={"text": "More?"}, headers=auth_headers)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
    assert "break" in limited.json()["detail"] and len(llm.stream_calls) == 1
    with app.state.sessionmaker() as db:  # the refused message left no half-made turn behind
        assert len(db.scalars(select(ConverseTurn).where(ConverseTurn.session_id == sid)).all()) == 1

    app.state.llm = UnavailableLLM()
    assert ask(client, auth_headers, sid)[-1][0] == "done"


# -------------------------------------------------------------------------------- checks


def test_perfect_check_promotes_listening_and_expressions(app, client, auth_headers):
    llm = FakeLLM(chunks=[ANSWER], check=CHECK)
    app.state.llm = llm
    sid = start(client, auth_headers)["session_id"]
    turn_id = ask(client, auth_headers, sid)[0][1]["turn_id"]

    response = check(client, auth_headers, sid, turn_id)
    assert response.status_code == 200, response.text
    body = response.json()
    questions = body["questions"]
    assert [question["kind"] for question in questions] == ["main_idea", "detail", "emotion"]
    assert all(set(question) == {"question_id", "kind", "prompt", "choices"} for question in questions)
    emotion = questions[2]
    assert emotion["prompt"].startswith("How did the avatar feel when it said “")
    assert len(emotion["choices"]) == 3  # emotion_choices at level 0

    result = submit(client, auth_headers, body["check_id"], answer_key(app, body["check_id"]))
    assert result.status_code == 200, result.text
    outcome = result.json()
    assert outcome["score"] == 1.0 and outcome["perfect"] is True
    assert all(item["correct"] and item["explanation"] for item in outcome["results"])
    assert [(p["track"], p["promoted"], p["level"]) for p in outcome["progressions"]] == [
        ("converse.listening", True, 1),
        ("converse.expressions", True, 1),
    ]
    tracks = {track["key"]: track for track in client.get("/api/progress", headers=auth_headers).json()["tracks"]}
    assert tracks["converse.listening"]["level"] == 1 and tracks["converse.expressions"]["level"] == 1
    assert start(client, auth_headers)["words_per_minute"] > 110.0


def test_wrong_answers_hold_the_level_and_explain(app, client, auth_headers):
    sid = start(client, auth_headers)["session_id"]  # demo guide and offline questions
    turn_id = ask(client, auth_headers, sid, "How many hearts does an octopus have?")[0][1]["turn_id"]
    body = check(client, auth_headers, sid, turn_id).json()
    key = answer_key(app, body["check_id"])
    wrong = {q["question_id"]: next(c for c in q["choices"] if c != key[q["question_id"]]) for q in body["questions"]}

    outcome = submit(client, auth_headers, body["check_id"], wrong).json()
    assert outcome["score"] == 0.0 and outcome["perfect"] is False
    assert all(not item["correct"] and item["expected"] == key[item["question_id"]] for item in outcome["results"])
    assert all(item["explanation"] for item in outcome["results"])
    assert {p["track"]: p["reason"] for p in outcome["progressions"]} == {
        "converse.listening": "held",
        "converse.expressions": "held",
    }


def test_all_neutral_answer_only_scores_listening(app, client, auth_headers):
    neutral = "[neutral] An octopus has three hearts in its body. [neutral] Two hearts pump blood to the gills."
    app.state.llm = FakeLLM(chunks=[neutral], check=CHECK)
    sid = start(client, auth_headers)["session_id"]
    turn_id = ask(client, auth_headers, sid)[0][1]["turn_id"]
    body = check(client, auth_headers, sid, turn_id).json()
    assert [question["kind"] for question in body["questions"]] == ["main_idea", "detail"]
    outcome = submit(client, auth_headers, body["check_id"], answer_key(app, body["check_id"])).json()
    assert [p["track"] for p in outcome["progressions"]] == ["converse.listening"]


def test_unanswered_questions_count_as_wrong(app, client, auth_headers):
    app.state.llm = FakeLLM(chunks=[ANSWER], check=CHECK)
    sid = start(client, auth_headers)["session_id"]
    turn_id = ask(client, auth_headers, sid)[0][1]["turn_id"]
    body = check(client, auth_headers, sid, turn_id).json()
    key = answer_key(app, body["check_id"])
    outcome = submit(client, auth_headers, body["check_id"], {"q1": key["q1"], "bogus": "x"}).json()
    assert [item["correct"] for item in outcome["results"]] == [True, False, False]
    assert outcome["score"] == pytest.approx(1 / 3, abs=1e-4)


def test_a_check_is_reused_until_submitted_and_scored_once(app, client, auth_headers):
    llm = FakeLLM(chunks=[ANSWER], check=CHECK)
    app.state.llm = llm
    sid = start(client, auth_headers)["session_id"]
    turn_id = ask(client, auth_headers, sid)[0][1]["turn_id"]
    first = check(client, auth_headers, sid, turn_id).json()
    assert check(client, auth_headers, sid, turn_id).json() == first
    assert len(llm.structured_calls) == 1

    key = answer_key(app, first["check_id"])
    assert submit(client, auth_headers, first["check_id"], key).status_code == 200
    assert submit(client, auth_headers, first["check_id"], key).status_code == 409
    assert check(client, auth_headers, sid, turn_id).status_code == 409
    tracks = {track["key"]: track for track in client.get("/api/progress", headers=auth_headers).json()["tracks"]}
    assert tracks["converse.listening"]["sessions"] == 1


def test_checks_belong_to_their_learner_session_and_turn(app, client, auth_headers):
    sid = start(client, auth_headers)["session_id"]
    turn_id = ask(client, auth_headers, sid)[0][1]["turn_id"]
    stranger = other_user(client)
    other_sid = start(client, auth_headers)["session_id"]

    assert check(client, stranger, sid, turn_id).status_code == 404
    assert check(client, auth_headers, other_sid, turn_id).status_code == 404  # turn of another session
    assert check(client, auth_headers, sid, "missing").status_code == 404

    client.post(f"/api/converse/sessions/{sid}/end", headers=auth_headers)
    body = check(client, auth_headers, sid, turn_id)  # the check comes after the conversation
    assert body.status_code == 200
    check_id = body.json()["check_id"]
    assert submit(client, stranger, check_id, {}).status_code == 404
    assert submit(client, auth_headers, "missing", {}).status_code == 404


def test_claude_checks_are_rate_limited(app, client, auth_headers, monkeypatch):
    monkeypatch.setattr(ratelimit.llm_limiter, "limit", 1)
    app.state.llm = FakeLLM(chunks=[ANSWER], check=CHECK)
    sid = start(client, auth_headers)["session_id"]
    turn_id = ask(client, auth_headers, sid)[0][1]["turn_id"]  # uses the one Claude call
    limited = check(client, auth_headers, sid, turn_id)
    assert limited.status_code == 429 and "Retry-After" in limited.headers

    app.state.llm = UnavailableLLM()  # offline questions cost nothing
    assert check(client, auth_headers, sid, turn_id).status_code == 200
