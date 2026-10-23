"""Math-fact rounds over HTTP: round → submit → progress, plus ownership and validation errors."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from legible.api.models import FactMemory, PracticeSession, TrackStateRow
from legible.progression import TRACKS

from ..conftest import register

TRACK = "numeracy.fluency"
SPEC = TRACKS[TRACK]
ROUND_KEYS = {"round_id", "track", "skill", "skill_label", "level", "time_limit_ms", "parameters", "problems"}
PROBLEM_KEYS = {"problem_id", "kind", "prompt", "visual", "answer_format", "choices"}


def new_round(client, headers) -> dict:
    response = client.get("/api/numeracy/round", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def submit(client, headers, round_id, answers):
    return client.post(f"/api/numeracy/round/{round_id}/submit", json={"answers": answers}, headers=headers)


def answer_key(app, round_id) -> list[dict]:
    with app.state.sessionmaker() as db:
        return db.get(PracticeSession, round_id).payload["problems"]


def right_answers(app, round_id, latency_ms=900) -> list[dict]:
    return [
        {"problem_id": p["problem_id"], "answer": str(p["answer"]), "latency_ms": latency_ms}
        for p in answer_key(app, round_id)
    ]


def user_id(client, headers) -> str:
    return client.get("/api/auth/me", headers=headers).json()["id"]


def set_level(app, uid, level) -> None:
    with app.state.sessionmaker() as db:
        db.add(TrackStateRow(user_id=uid, track_key=TRACK, level=level))
        db.commit()


def numeracy_track(client, headers) -> dict:
    tracks = client.get("/api/progress", headers=headers).json()["tracks"]
    return next(track for track in tracks if track["key"] == TRACK)


def test_a_new_round_matches_the_contract_and_hides_answers(client, auth_headers):
    body = new_round(client, auth_headers)
    assert set(body) == ROUND_KEYS
    assert body["track"] == TRACK and body["level"] == 0
    assert body["skill"] == "subitize_5" and body["skill_label"] == "Seeing amounts up to 5"
    assert body["time_limit_ms"] == round(10_000 * 0.6)
    assert set(body["parameters"]) == {"time_limit_ms", "problems", "visual_support"}
    assert body["parameters"]["problems"]["value"] == 8 and len(body["problems"]) == 8
    for problem in body["problems"]:
        assert set(problem) == PROBLEM_KEYS
        assert problem["kind"] == "subitize" and problem["answer_format"] == "integer"
        assert problem["visual"]["type"] == "dots" and problem["visual"]["layout"] in {"dice", "random", "line"}


def test_a_perfect_round_levels_up_and_shows_in_progress(app, client, auth_headers):
    body = new_round(client, auth_headers)
    response = submit(client, auth_headers, body["round_id"], right_answers(app, body["round_id"]))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["score"] == 1.0 and result["perfect"] is True
    assert result["fluent_count"] == result["total"] == 8
    assert [r["problem_id"] for r in result["results"]] == [p["problem_id"] for p in body["problems"]]
    assert all(r["correct"] and r["fluent"] and r["strategy_hint"] is None for r in result["results"])
    (update,) = result["progressions"]
    assert update["track"] == TRACK and update["promoted"] and update["level"] == 1 and update["reason"] == "perfect"

    track = numeracy_track(client, auth_headers)
    today = datetime.now(timezone.utc).date().isoformat()
    assert track["level"] == 1 and track["sessions"] == 1 and track["perfect_sessions"] == 1
    assert track["history"] == [{"date": today, "level": 1, "score": 1.0}]

    following = new_round(client, auth_headers)
    assert following["level"] == 1 and following["time_limit_ms"] < body["time_limit_ms"]


def test_accurate_but_slow_rounds_hold_the_level_while_wrong_ones_step_back(app, client, auth_headers):
    set_level(app, user_id(client, auth_headers), 30)
    for _ in range(3):
        body = new_round(client, auth_headers)
        slow = right_answers(app, body["round_id"], latency_ms=body["time_limit_ms"] + 1000)
        result = submit(client, auth_headers, body["round_id"], slow).json()
        assert result["score"] == 0.7 and not result["perfect"] and result["fluent_count"] == 0
        assert all(r["correct"] and not r["fluent"] and r["strategy_hint"] for r in result["results"])
        assert result["progressions"][0]["reason"] == "held" and result["progressions"][0]["level"] == 30

    reasons = []
    for _ in range(3):
        body = new_round(client, auth_headers)
        wrong = [{"problem_id": p["problem_id"], "answer": "-1", "latency_ms": 900} for p in body["problems"]]
        result = submit(client, auth_headers, body["round_id"], wrong).json()
        assert result["score"] == 0 and not any(r["correct"] for r in result["results"])
        reasons.append(result["progressions"][0]["reason"])
    assert reasons == ["held", "held", "struggling"]
    assert numeracy_track(client, auth_headers)["level"] == 29


def test_wrong_and_missing_answers_show_the_answer_and_a_strategy(app, client, auth_headers):
    body = new_round(client, auth_headers)
    key = answer_key(app, body["round_id"])
    answers = right_answers(app, body["round_id"])
    answers[0]["answer"] = "99"
    del answers[1]
    result = submit(client, auth_headers, body["round_id"], answers).json()

    assert result["score"] == pytest.approx((len(key) - 2) / len(key), abs=1e-4) and not result["perfect"]
    wrong, missing, *rest = result["results"]
    for graded, stored in ((wrong, key[0]), (missing, key[1])):
        assert not graded["correct"] and not graded["fluent"]
        assert graded["expected"] == str(stored["answer"]) and graded["strategy_hint"] == stored["hint"]
    assert all(r["correct"] and r["strategy_hint"] is None for r in rest)
    assert result["progressions"][0]["reason"] == "held"


def test_each_answer_updates_the_learners_fact_memory(app, client, auth_headers):
    body = new_round(client, auth_headers)
    key = answer_key(app, body["round_id"])
    answers = right_answers(app, body["round_id"])
    answers[1]["latency_ms"] = 20_000
    answers[2]["answer"] = "0"
    submit(client, auth_headers, body["round_id"], answers)

    uid = user_id(client, auth_headers)
    with app.state.sessionmaker() as db:
        rows = {row.fact: row for row in db.scalars(select(FactMemory).where(FactMemory.user_id == uid))}
    assert set(rows) == {p["fact"] for p in key}
    quick, slow, wrong = rows[key[0]["fact"]], rows[key[1]["fact"]], rows[key[2]["fact"]]
    assert (quick.seen, quick.correct, quick.fluent, quick.mean_latency_ms) == (1, 1, 1, 900)
    assert (slow.seen, slow.correct, slow.fluent, slow.mean_latency_ms) == (1, 1, 0, 20_000)
    assert (wrong.seen, wrong.correct, wrong.fluent, wrong.mean_latency_ms) == (1, 0, 0, None)
    assert all(row.skill == "subitize_5" and row.last_seen_at is not None for row in rows.values())


def test_typed_answers_are_read_leniently(app, client, auth_headers):
    body = new_round(client, auth_headers)
    answers = right_answers(app, body["round_id"])
    answers[0]["answer"] = int(answers[0]["answer"])  # a bare JSON number
    answers[1]["answer"] = f"  {answers[1]['answer']} "
    answers[2]["answer"] = chr(0xFF10 + int(answers[2]["answer"]))  # a full-width digit
    result = submit(client, auth_headers, body["round_id"], answers).json()
    assert result["perfect"], result


def test_compare_problems_are_answered_with_one_of_their_choices(app, client, auth_headers):
    set_level(app, user_id(client, auth_headers), 24)  # "Which number is more?"
    body = new_round(client, auth_headers)
    assert body["skill"] == "compare_10"
    choice_ids = [p["problem_id"] for p in body["problems"] if p["answer_format"] == "choice"]
    assert choice_ids and all(len(p["choices"]) == 2 for p in body["problems"] if p["kind"] == "compare")
    answers = right_answers(app, body["round_id"])
    not_a_choice = next(a for a in answers if a["problem_id"] == choice_ids[0])
    not_a_choice["answer"] = "11"
    results = {r["problem_id"]: r for r in submit(client, auth_headers, body["round_id"], answers).json()["results"]}
    assert not results[choice_ids[0]]["correct"]
    assert all(results[pid]["correct"] for pid in choice_ids[1:])


def test_weak_facts_from_earlier_skills_come_back_in_review(app, client, auth_headers):
    uid = user_id(client, auth_headers)
    set_level(app, uid, 60)  # pairs that make 10: 9 problems, 3 of them review
    weak = {"compare:3:8": "compare_10", "subitize:dice:4": "subitize_5", "line:10:7": "number_line_10"}
    with app.state.sessionmaker() as db:
        for fact, skill in weak.items():
            db.add(FactMemory(user_id=uid, fact=fact, skill=skill, seen=4, correct=1, fluent=0, mean_latency_ms=9000,
                              last_seen_at=datetime.now(timezone.utc)))
        db.commit()
    body = new_round(client, auth_headers)
    review = {p["fact"] for p in answer_key(app, body["round_id"]) if p["review"]}
    assert review == set(weak)


def test_a_round_follows_the_learners_level(app, client, auth_headers):
    set_level(app, user_id(client, auth_headers), 200)
    body = new_round(client, auth_headers)
    values = SPEC.parameter_values(200)
    assert body["skill"] == "mul_10" and body["level"] == 200
    assert len(body["problems"]) == int(values["problems"]) == 13
    assert body["time_limit_ms"] == round(values["time_limit_ms"] * 1.3)
    assert sum(p["kind"] == "mul" for p in body["problems"]) >= 9


def test_rounds_belong_to_their_learner(app, client, auth_headers):
    body = new_round(client, auth_headers)
    answers = right_answers(app, body["round_id"])
    other = {"Authorization": f"Bearer {register(client)['access_token']}", "X-Timezone": "UTC"}
    assert submit(client, other, body["round_id"], answers).status_code == 404
    assert submit(client, auth_headers, "not-a-round", answers).status_code == 404
    assert submit(client, auth_headers, body["round_id"], answers).status_code == 200


def test_a_round_can_only_be_submitted_once(app, client, auth_headers):
    body = new_round(client, auth_headers)
    answers = right_answers(app, body["round_id"])
    assert submit(client, auth_headers, body["round_id"], answers).status_code == 200
    again = submit(client, auth_headers, body["round_id"], answers)
    assert again.status_code == 409 and again.json()["detail"]
    assert numeracy_track(client, auth_headers)["sessions"] == 1


def test_unknown_or_repeated_problem_ids_are_rejected_without_side_effects(app, client, auth_headers):
    body = new_round(client, auth_headers)
    answers = right_answers(app, body["round_id"])
    unknown = [*answers, {"problem_id": "elsewhere", "answer": "1", "latency_ms": 900}]
    repeated = [*answers, answers[0]]
    for bad in (unknown, repeated):
        response = submit(client, auth_headers, body["round_id"], bad)
        assert response.status_code == 422 and isinstance(response.json()["detail"], str)
    with app.state.sessionmaker() as db:
        assert db.scalars(select(FactMemory)).first() is None
    assert numeracy_track(client, auth_headers)["sessions"] == 0
    assert submit(client, auth_headers, body["round_id"], answers).status_code == 200


def test_malformed_answers_are_rejected(client, auth_headers):
    round_id = new_round(client, auth_headers)["round_id"]
    problem_id = "x"
    for bad in (
        {"answers": [{"problem_id": problem_id, "answer": "1", "latency_ms": -5}]},
        {"answers": [{"problem_id": problem_id, "answer": "1" * 40, "latency_ms": 900}]},
        {"answers": [{"problem_id": problem_id, "latency_ms": 900}]},
        {},
    ):
        response = client.post(f"/api/numeracy/round/{round_id}/submit", json=bad, headers=auth_headers)
        assert response.status_code == 422


def test_a_round_left_open_overnight_still_counts_as_slow(app, client, auth_headers):
    body = new_round(client, auth_headers)
    three_days = 3 * 24 * 60 * 60 * 1000
    response = submit(client, auth_headers, body["round_id"], right_answers(app, body["round_id"], three_days))
    assert response.status_code == 200
    assert response.json()["score"] == 0.7 and response.json()["fluent_count"] == 0


def test_rounds_require_login(client):
    assert client.get("/api/numeracy/round").status_code == 401
    assert client.post("/api/numeracy/round/abc/submit", json={"answers": []}).status_code == 401
