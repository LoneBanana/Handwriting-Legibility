from datetime import date, timedelta

import pytest

from legible.progression import TRACKS, Curve, TrackSpec, TrackState, apply_session

SPEC = TrackSpec(key="t", module="literacy", label="T", description="", levels_to_target=10)
DAY = date(2026, 1, 5)


def run(scores, start=TrackState(), spec=SPEC, day=DAY, step_days=1):
    state, outcomes = start, []
    for i, score in enumerate(scores):
        state, outcome = apply_session(spec, state, score, day + timedelta(days=i * step_days))
        outcomes.append(outcome)
    return state, outcomes


def test_perfect_session_promotes_once():
    state, (outcome,) = run([1.0])
    assert state.level == 1 and outcome.promoted and outcome.reason == "perfect"
    assert state.streak == 1 and state.perfect_sessions == 1 and state.last_promoted_on == DAY


def test_imperfect_session_holds():
    state, (outcome,) = run([0.9])
    assert state.level == 0 and outcome.reason == "held" and not outcome.promoted
    assert state.streak == 0 and state.sessions == 1


def test_daily_cap_blocks_second_promotion_same_day():
    state, outcomes = run([1.0, 1.0, 1.0], step_days=0)
    assert state.level == 1
    assert [o.reason for o in outcomes] == ["perfect", "daily_cap", "daily_cap"]
    assert state.streak == 3 and state.best_streak == 3


def test_next_day_promotes_again():
    state, outcomes = run([1.0, 1.0])
    assert state.level == 2 and [o.reason for o in outcomes] == ["perfect", "perfect"]


def test_struggle_window_steps_back_one_level():
    start = TrackState(level=5)
    state, outcomes = run([0.3, 0.2, 0.5], start=start)
    assert [o.reason for o in outcomes] == ["held", "held", "struggling"]
    assert state.level == 4 and outcomes[-1].demoted and state.struggle_count == 0


def test_middling_session_breaks_struggle_run():
    state, outcomes = run([0.3, 0.3, 0.7, 0.3, 0.3], start=TrackState(level=5))
    assert state.level == 5 and all(o.reason == "held" for o in outcomes)


def test_struggling_at_level_zero_never_goes_negative():
    state, outcomes = run([0.0, 0.0, 0.0])
    assert state.level == 0 and outcomes[-1].reason == "held"


def test_max_level():
    state, (outcome,) = run([1.0], start=TrackState(level=10))
    assert state.level == 10 and outcome.reason == "max_level"


def test_scores_are_clamped():
    _, (outcome,) = run([float("nan")])
    assert outcome.score == 0.0
    _, (outcome,) = run([7.0])
    assert outcome.score == 1.0 and outcome.promoted


def test_geometric_curve_endpoints_and_constant_ratio():
    curve = Curve(110, 180)
    assert curve.at(0, 300) == pytest.approx(110)
    assert curve.at(300, 300) == pytest.approx(180)
    assert curve.at(1000, 300) == pytest.approx(180)
    ratio = curve.step_ratio(300)
    assert curve.at(101, 300) / curve.at(100, 300) == pytest.approx(ratio)
    assert ratio - 1 < 0.002  # far below any noticeable day-to-day change


def test_decreasing_geometric_and_integer_linear_curves():
    assert Curve(10000, 2500).at(150, 300) == pytest.approx(5000)
    hints = Curve(2, 0, shape="linear", integer=True)
    values = [hints.at(level, 300) for level in range(0, 301, 30)]
    assert values[0] == 2 and values[-1] == 0
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("key", sorted(TRACKS))
def test_registered_tracks_describe_parameters(key):
    spec = TRACKS[key]
    assert spec.key == key
    for level in (0, spec.levels_to_target // 2, spec.levels_to_target):
        params = spec.describe_parameters(level)
        assert params, key
        for info in params.values():
            lo, hi = sorted((info["start"], info["target"]))
            assert lo - 1e-9 <= info["value"] <= hi + 1e-9
