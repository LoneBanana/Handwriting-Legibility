from datetime import datetime, timedelta, timezone

import pytest

from legible.literacy import MemoryState, analyze, phonetically_plausible, recall_probability, update
from legible.literacy.distractors import BLOCKED, distractors


@pytest.mark.parametrize(
    ("expected", "response", "types"),
    [
        ("dog", "bog", ["reversal"]),
        ("pen", "qen", ["reversal"]),
        ("form", "from", ["transposition"]),
        ("hill", "hil", ["doubling"]),
        ("hop", "hopp", ["doubling"]),
        ("cat", "cot", ["vowel"]),
        ("stop", "sop", ["omission"]),
        ("cat", "cart", ["insertion"]),
        ("ship", "chip", ["substitution"]),
        ("cat", "Cat.", []),
    ],
)
def test_error_types(expected, response, types):
    assert [e.type for e in analyze(expected, response)] == types


def test_errors_have_positions_and_plain_details():
    (error,) = analyze("bed", "ded")
    assert error.position == 0 and "mirror" in error.detail


@pytest.mark.parametrize(
    ("expected", "response", "plausible"),
    [
        ("phone", "fone", True), ("night", "nite", True), ("boat", "bote", True), ("rain", "rane", True),
        ("jumped", "jumpt", True), ("cat", "kat", True), ("bell", "bel", True), ("knife", "nife", True),
        ("write", "rite", True), ("ship", "sip", False), ("bed", "bet", False), ("cat", "cat", False),
        ("cat", "", False),
    ],
)
def test_sounds_right(expected, response, plausible):
    assert phonetically_plausible(expected, response) is plausible


def test_memory_half_life_grows_and_shrinks():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = MemoryState()
    assert recall_probability(state, now) == 0.0
    state = update(state, True, 2000, now)
    assert state.half_life_days > 1.0 and recall_probability(state, now) == pytest.approx(1.0)
    later = now + timedelta(days=state.half_life_days)
    assert recall_probability(state, later) == pytest.approx(0.5)
    shrunk = update(state, False, 9000, later)
    assert shrunk.half_life_days < state.half_life_days and shrunk.wrong == 1 and shrunk.seen == 2


def test_distractors_are_distinct_confusable_and_clean():
    for word in ("dog", "duck", "phone", "ship", "bike", "night", "station"):
        picks = distractors(word, 4, seed=word)
        assert len(picks) == 4 and len(set(picks)) == 4 and word not in picks
        assert not set(picks) & BLOCKED
    assert distractors("duck", 4, "a") == distractors("duck", 4, "a")
    assert "dick" not in {p for seed in range(50) for p in distractors("duck", 4, str(seed))}


def test_messages_after_a_transposition_still_name_the_learner():
    errors = analyze("formed", "fromid")
    assert errors[0].type == "transposition"
    assert all(e.detail.startswith(("“", "You wrote", "The vowel", "The ", "There", "This")) for e in errors)
    picked = analyze("ham", "hat", chosen=True)
    assert picked[0].detail.startswith("Your choice has")
