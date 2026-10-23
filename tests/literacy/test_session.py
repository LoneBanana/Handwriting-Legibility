import random
from datetime import datetime, timedelta, timezone

from legible.literacy import SKILLS, MemoryState, build_session, check_answer, word_bank

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_first_session_matches_level_zero_parameters():
    plan = build_session(0, 0, {}, NOW, random.Random(0))
    read = [c for c in plan.cards if c.mode == "read"]
    spell = [c for c in plan.cards if c.mode == "spell"]
    assert plan.skill.id == "short_a" and len(read) == 5 and len(spell) == 4
    assert {c.word for c in read}.isdisjoint({c.word for c in spell})
    for card in read:
        assert card.exposure_ms == 2000 and len(card.choices) == 3 and card.choices[0] == card.word
    for card in spell:
        assert card.hint == {"length": len(card.word), "first_letter": card.word[0]} and card.sentence
    assert [c.mode for c in plan.cards][:4] == ["read", "spell", "read", "spell"]


def test_later_sessions_mix_review_and_fade_support():
    plan = build_session(300, 300, {}, NOW, random.Random(1))
    assert plan.skill.id == SKILLS[-1].id
    assert len([c for c in plan.cards if c.mode == "read"]) == 8
    assert all(c.hint is None for c in plan.cards if c.mode == "spell")
    assert all(len(c.choices) == 5 and c.exposure_ms == 500 for c in plan.cards if c.mode == "read")
    skills = {c.skill for c in plan.cards}
    assert SKILLS[-1].id in skills and len(skills) > 1  # current pattern plus review


def test_review_prefers_words_likely_forgotten():
    words = [w for skill in SKILLS[:3] for w in word_bank()[skill.id]]
    memory = {(w.text, mode): MemoryState(seen=3, correct=3, half_life_days=30.0, last_seen_at=NOW)
              for w in words for mode in ("read", "spell")}
    memory[("hat", "read")] = MemoryState(seen=3, correct=1, wrong=2, half_life_days=0.5,
                                          last_seen_at=NOW - timedelta(days=5))
    plan = build_session(8 * 3, 8 * 3, memory, NOW, random.Random(2))
    assert "hat" in [c.word for c in plan.cards if c.mode == "read"]


def test_check_answer_feedback():
    plan = build_session(0, 0, {}, NOW, random.Random(0))
    spell = next(c for c in plan.cards if c.mode == "spell")
    right = check_answer(spell, spell.word.upper() + " ")
    assert right.correct and right.errors == [] and right.feedback
    wrong = check_answer(spell, spell.word[:-1])
    assert not wrong.correct and wrong.errors and wrong.expected == spell.word
    empty = check_answer(spell, "")
    assert not empty.correct and empty.errors == []
