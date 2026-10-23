import random
from datetime import date, datetime, timedelta, timezone

import pytest

from legible.numeracy import AnswerKey, FactStats, grade, parse_integer, priority, record_attempt, score_round
from legible.numeracy.memory import LATENCY_CAP_MS, need
from legible.numeracy.scoring import SLOW_CREDIT
from legible.progression import TRACKS, TrackState, apply_session

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
LIMIT = 4000


def key(answer=56, *, answer_format="integer", choices=None, tolerance=0, problem_id="p1"):
    return AnswerKey(
        problem_id=problem_id,
        skill="mul_7",
        fact="mul:7x8",
        answer_format=answer_format,
        answer=answer,
        tolerance=tolerance,
        choices=choices,
        hint="7 × 8: 8 × 5 is 40, plus 8 × 2 is 16, so 56.",
    )


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("7", 7),
        (" 12 ", 12),
        (" 15\t", 15),
        ("+4", 4),
        ("-3", -3),
        ("−3", -3),   # true minus sign
        ("–3", -3),   # en dash
        ("－3", -3),   # full-width hyphen-minus
        ("- 3", -3),
        ("０７", 7),       # full-width digits
        ("007", 7),
        ("", None),
        ("   ", None),
        ("7.0", None),
        ("1 2", None),
        ("seven", None),
        ("--3", None),
        ("3-", None),
    ],
)
def test_integer_answers_are_read_leniently(text, value):
    assert parse_integer(text) == value


def test_quick_and_correct_is_fluent_with_full_credit_and_no_hint():
    graded = grade(key(), "56", 1800, LIMIT)
    assert graded.correct and graded.fluent and graded.credit == 1.0 and graded.strategy_hint is None
    assert graded.expected == "56"
    assert grade(key(), " 56 ", LIMIT, LIMIT).fluent  # exactly at the limit still counts


def test_correct_but_slow_gets_partial_credit_and_a_strategy():
    graded = grade(key(), "56", LIMIT + 1, LIMIT)
    assert graded.correct and not graded.fluent and graded.credit == SLOW_CREDIT == 0.7
    assert graded.strategy_hint and "56" in graded.strategy_hint


def test_wrong_or_missing_answers_get_no_credit_and_a_strategy():
    wrong = grade(key(), "54", 900, LIMIT)
    missing = grade(key(), None, None, LIMIT)
    assert not wrong.correct and wrong.credit == 0 and wrong.strategy_hint
    assert not missing.answered and not missing.correct and missing.credit == 0 and missing.strategy_hint
    assert grade(key(), "abc", 900, LIMIT).credit == 0


def test_choice_answers_must_match_a_choice():
    compare = key("7", answer_format="choice", choices=("4", "7"))
    assert grade(compare, "7", 500, LIMIT).correct
    assert grade(compare, " 7 ", 500, LIMIT).correct
    assert grade(compare, "０７", 500, LIMIT).correct  # names the same number as a choice
    assert not grade(compare, "4", 500, LIMIT).correct
    assert not grade(compare, "9", 500, LIMIT).correct  # not one of the choices
    assert not grade(compare, "seven", 500, LIMIT).correct


def test_number_line_estimates_count_within_the_tolerance():
    line = key(37, tolerance=3)
    assert all(grade(line, str(answer), 900, LIMIT).correct for answer in (34, 36, 37, 40))
    assert not any(grade(line, str(answer), 900, LIMIT).correct for answer in (33, 41, 73))
    assert grade(line, "35", 900, LIMIT).expected == "37"  # the exact value is still reported


def test_only_an_all_quick_all_correct_round_is_perfect():
    def round_of(*outcomes):
        answers = {"fluent": ("56", 1000), "slow": ("56", 9000), "wrong": ("50", 1000), "missing": (None, None)}
        return score_round([grade(key(problem_id=str(i)), *answers[o], LIMIT) for i, o in enumerate(outcomes)])

    perfect = round_of("fluent", "fluent", "fluent", "fluent")
    assert perfect.perfect and perfect.score == 1.0 and perfect.fluent == 4
    one_slow = round_of("fluent", "fluent", "fluent", "slow")
    assert not one_slow.perfect and one_slow.score == pytest.approx((3 + 0.7) / 4)
    all_slow = round_of("slow", "slow", "slow", "slow")
    assert not all_slow.perfect and all_slow.score == pytest.approx(0.7) and all_slow.correct == 4
    mixed = round_of("fluent", "wrong", "missing", "slow")
    assert mixed.score == pytest.approx(1.7 / 4) and mixed.correct == 2 and mixed.fluent == 1
    assert score_round([]).score == 0 and not score_round([]).perfect


def test_accurate_but_slow_rounds_never_count_as_struggling():
    spec = TRACKS["numeracy.fluency"]
    assert SLOW_CREDIT >= spec.struggle_threshold
    state = TrackState(level=40)
    for day in range(6):
        state, outcome = apply_session(spec, state, 0.7, date(2026, 1, 5) + timedelta(days=day))
        assert outcome.reason == "held"
    assert state.level == 40


def test_recording_an_attempt_updates_counts_speed_and_time():
    stats = record_attempt(FactStats(), correct=True, fluent=True, latency_ms=2000, now=NOW)
    assert (stats.seen, stats.correct, stats.fluent, stats.mean_latency_ms, stats.last_seen_at) == (1, 1, 1, 2000, NOW)
    stats = record_attempt(stats, correct=True, fluent=False, latency_ms=6000, now=NOW)
    assert stats.seen == 2 and stats.correct == 2 and stats.fluent == 1
    assert 2000 < stats.mean_latency_ms < 6000  # weighted towards recent answers
    wrong = record_attempt(stats, correct=False, fluent=False, latency_ms=500, now=NOW)
    assert wrong.seen == 3 and wrong.correct == 2 and wrong.mean_latency_ms == stats.mean_latency_ms
    away = record_attempt(FactStats(), correct=True, fluent=False, latency_ms=10 * 60_000, now=NOW)
    assert away.mean_latency_ms == LATENCY_CAP_MS


def test_priority_puts_weak_then_new_then_known_facts():
    def rank(stats, seed=0):
        return priority(stats, now=NOW, time_limit_ms=LIMIT, rng=random.Random(seed))

    weak = FactStats(4, 1, 0, 3000.0, NOW)
    slow = FactStats(4, 4, 0, 9000.0, NOW)
    known_now = FactStats(6, 6, 6, 900.0, NOW)
    known_long_ago = FactStats(6, 6, 6, 900.0, NOW - timedelta(days=30))
    for seed in range(20):
        assert rank(weak, seed) > 1.5 and rank(slow, seed) > 1.5
        assert 1.0 <= rank(None, seed) <= 1.5
        assert rank(known_now, seed) <= 0.5
        assert 1.0 <= rank(known_long_ago, seed) <= 1.5  # due again after a gap
    assert need(known_now, LIMIT) == 0 and need(FactStats(2, 0, 0), LIMIT) == 5


def test_priority_reads_naive_database_times_as_utc():
    naive = FactStats(6, 6, 6, 900.0, (NOW - timedelta(days=30)).replace(tzinfo=None))
    assert priority(naive, now=NOW, time_limit_ms=LIMIT, rng=random.Random(0)) >= 1.0
