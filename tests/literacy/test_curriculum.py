import re
from collections import Counter

import pytest

from legible.literacy import LEVELS_PER_SKILL, SKILLS, SKILLS_BY_ID, focus_span, skill_index_for, word_bank


def test_every_skill_has_enough_words():
    bank = word_bank()
    assert [s.id for s in SKILLS] == list(bank)
    for skill in SKILLS:
        assert len(bank[skill.id]) >= 9, skill.id


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.id)
def test_rows_are_consistent(skill):
    for word in word_bank()[skill.id]:
        assert "".join(word.graphemes) == word.text and all(word.graphemes), word
        assert re.fullmatch(r"[a-z]+", word.text), word
        assert re.search(rf"\b{word.text}\b", word.sentence, re.IGNORECASE), word
        assert word.sentence[0].isupper() and word.sentence[-1] in ".?!", word
        if skill.id != "two_syllable":
            start, end = word.focus
            assert 0 <= start < end <= len(word.text)


def test_words_are_unique_across_skills():
    counts = Counter(w.text for words in word_bank().values() for w in words)
    assert [w for w, n in counts.items() if n > 1] == []


def test_focus_spans():
    assert focus_span("cake", ("c", "a", "k", "e"), SKILLS_BY_ID["vce_a"]) == (1, 4)
    assert focus_span("ship", ("sh", "i", "p"), SKILLS_BY_ID["digraph_sh"]) == (0, 2)
    assert focus_span("jumping", ("j", "u", "m", "p", "i", "ng"), SKILLS_BY_ID["ending_ing_ed"]) == (4, 7)
    # The "a" inside the vowel team "ai" is not a short a: patterns must align with graphemes.
    assert focus_span("rain", ("r", "ai", "n"), SKILLS_BY_ID["short_a"]) is None


def test_new_patterns_need_both_reading_and_spelling():
    assert skill_index_for(0, 0) == 0
    assert skill_index_for(LEVELS_PER_SKILL, 0) == 0
    assert skill_index_for(LEVELS_PER_SKILL, LEVELS_PER_SKILL) == 1
    assert skill_index_for(10_000, 10_000) == len(SKILLS) - 1
