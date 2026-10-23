import json
import random
from datetime import datetime, timedelta, timezone

from legible.numeracy import LADDER, LEVELS_PER_SKILL, SKILLS, FactStats, plan_round, skill_index
from legible.numeracy.rounds import review_count
from legible.progression import TRACKS

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
TRACK = TRACKS["numeracy.fluency"]
CONTRACT_KEYS = {"problem_id", "kind", "prompt", "visual", "answer_format", "choices"}


def plan(level, memory=None, seed=0):
    return plan_round(level, memory or {}, rng=random.Random(seed), now=NOW)


def level_of(skill_id):
    return [skill.id for skill in LADDER].index(skill_id) * LEVELS_PER_SKILL


def history(seen, correct, fluent, latency_ms=1500.0, days_ago=1.0):
    return FactStats(seen, correct, fluent, latency_ms, NOW - timedelta(days=days_ago))


def test_rounds_at_every_level_follow_the_track():
    for level in range(0, TRACK.top_level + 1):
        for seed in (0, 1):
            round_ = plan(level, seed=seed)
            values = TRACK.parameter_values(level)
            index = skill_index(level)
            assert round_.skill is LADDER[index]
            assert len(round_.problems) == int(values["problems"])
            assert round_.time_limit_ms == round(values["time_limit_ms"] * round_.skill.time_factor)

            keys = [problem.fact.key for problem in round_.problems]
            assert len(set(keys)) == len(keys), f"duplicate fact at level {level}"
            assert len({problem.problem_id for problem in round_.problems}) == len(keys)

            reviews = [problem for problem in round_.problems if problem.review]
            assert len(reviews) == review_count(len(keys), index > 0)
            earlier = {skill.id for skill in LADDER[:index]}
            assert all(problem.skill in earlier for problem in reviews)
            assert all(problem.skill == round_.skill.id for problem in round_.problems if not problem.review)


def test_about_30_percent_of_a_round_is_review():
    assert review_count(8, True) == 2 and review_count(10, True) == 3 and review_count(15, True) == 5
    assert review_count(15, False) == 0
    assert all(0.25 <= review_count(n, True) / n <= 0.34 for n in range(8, 16))


def test_review_is_shuffled_into_the_round():
    positions = set()
    for seed in range(20):
        round_ = plan(level_of("mul_7"), seed=seed)
        positions |= {i for i, problem in enumerate(round_.problems) if problem.review}
    assert len(positions) > len(round_.problems) // 2


def test_pictures_fade_with_visual_support():
    assert all(problem.visual is not None for problem in plan(0).problems)
    for seed in range(5):  # visual support is 0% at the top level: only essential pictures remain
        for problem in plan(TRACK.top_level, seed=seed).problems:
            assert (problem.visual is not None) == problem.fact.needs_picture
    shown = optional = 0
    for seed in range(40):  # 50% support at level 150
        for problem in plan(150, seed=seed).problems:
            if not problem.fact.needs_picture:
                optional += 1
                shown += problem.visual is not None
    assert 0.4 < shown / optional < 0.6


def test_review_brings_back_the_weakest_earlier_facts():
    level = level_of("mul_7")  # 14 problems, 4 of them review
    weak = {
        "add:7+8": history(4, 1, 0),
        "sub:13-5": history(5, 2, 1),
        "bond:10:3+?": history(3, 0, 0),
        "line:100:37": history(4, 1, 1),
    }
    secure = {
        fact.key: history(6, 6, 6, latency_ms=900)
        for skill in LADDER[: skill_index(level)]
        for fact in skill.facts
        if fact.key not in weak
    }
    for seed in range(25):
        round_ = plan(level, {**secure, **weak}, seed)
        assert {problem.fact.key for problem in round_.problems if problem.review} == set(weak)


def test_slow_but_accurate_facts_are_reviewed_before_secure_ones():
    level = level_of("bonds_10")
    slow = {"compare:3:8": FactStats(4, 4, 0, 15_000.0, NOW)}
    secure = {
        fact.key: history(5, 5, 5, latency_ms=800, days_ago=0)
        for skill in LADDER[: skill_index(level)]
        for fact in skill.facts
        if fact.key not in slow
    }
    for seed in range(25):
        round_ = plan(level, {**secure, **slow}, seed)
        assert "compare:3:8" in {problem.fact.key for problem in round_.problems if problem.review}


def test_new_facts_come_before_ones_just_mastered():
    skill = SKILLS["add_sub_10"]
    mastered = {fact.key: FactStats(5, 5, 5, 1000.0, NOW) for fact in skill.facts[:-8]}
    for seed in range(10):
        problems = plan(level_of("add_sub_10"), mastered, seed).problems
        assert not {problem.fact.key for problem in problems if not problem.review} & set(mastered)


def test_weak_facts_of_the_current_skill_come_up_first():
    level = level_of("sub_20")
    weak = {"sub:13-5": history(3, 0, 0), "sub:16-9": history(4, 1, 0)}
    for seed in range(10):
        assert set(weak) <= {problem.fact.key for problem in plan(level, weak, seed).problems}


def test_the_client_view_never_includes_answers():
    round_ = plan(level_of("compare_10") + 5)
    for problem in round_.problems:
        public = problem.public()
        assert set(public) == CONTRACT_KEYS
    payload = json.loads(json.dumps(round_.to_payload()))  # stored as JSON
    assert payload["time_limit_ms"] == round_.time_limit_ms and payload["skill"] == "compare_10"
    for stored, problem in zip(payload["problems"], round_.problems):
        assert stored["answer"] == problem.fact.answer and stored["fact"] == problem.fact.key
        assert stored["hint"] == problem.hint and stored["skill"] == problem.skill
    assert any(stored["answer_format"] == "choice" for stored in payload["problems"])


def test_same_seed_same_round():
    first, second = plan(170, seed=7), plan(170, seed=7)
    assert [p.public() for p in first.problems] == [p.public() for p in second.problems]
