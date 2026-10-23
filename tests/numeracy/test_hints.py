import random
import re

import pytest

from legible.numeracy import LADDER, SKILLS, build_problem
from legible.numeracy import strategies as s
from legible.numeracy.facts import Add, Bond, Mul, NumberLinePoint, Sub, Subitize

M = "−"


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        (s.add(9, 6), "9 + 6: move 1 from the 6 to make 10, then 10 + 5 = 15."),
        (s.add(5, 8), "5 + 8: move 2 from the 5 to make 10, then 10 + 3 = 13."),
        (s.add(6, 7), "6 + 7: double 6 is 12, and 1 more is 13."),
        (s.add(8, 9, prefer_make_ten=True), "8 + 9: move 1 from the 8 to make 10, then 10 + 7 = 17."),
        (s.add(7, 7), "7 + 7: double 5 is 10 and double 2 is 4, so 14."),
        (s.add(2, 6), "2 + 6: start at 6 and count on 2: 7, 8."),
        (s.add(3, 7), "3 + 7: 3 and 7 make 10, like a full ten-frame."),
        (s.add(10, 4), "10 + 4: 1 ten and 4 ones make 14."),
        (s.subtract(9, 6), f"9 {M} 6: count up from 6 to 9: 7, 8, 9. That is 3."),
        (s.subtract(7, 2), f"7 {M} 2: count back 2 from 7: 6, 5."),
        (s.subtract(10, 7), f"10 {M} 7: 7 and 3 make 10, so 3 are left."),
        (s.subtract(15, 7), f"15 {M} 7: take 5 to get to 10, then take 2 more to get to 8."),
        (s.subtract(13, 9), f"13 {M} 9: count up from 9. 1 gets you to 10 and 3 more gets you to 13, so 4."),
        (s.subtract(17, 5), f"17 {M} 5: take 5 from the 7 in 17 to leave 2, so 12."),
        (s.multiply(9, 7), "9 × 7: 7 × 10 is 70, take away one 7 to get 63."),
        (s.multiply(5, 7), "5 × 7: half of 7 × 10. Half of 70 is 35."),
        (s.multiply(7, 2), "7 × 2: double 7 is 7 + 7 = 14."),
        (s.multiply(4, 6), "4 × 6: double 6 is 12, and double again is 24."),
        (s.multiply(7, 8), "7 × 8: 8 × 5 is 40, plus 8 × 2 is 16, so 56."),
        (s.multiply(8, 8), "8 × 8: 8 × 4 is 32, and double that is 64."),
        (s.divide(56, 8), "56 ÷ 8: think 8 × ? = 56. 8 × 7 = 56, so 7."),
        (s.bond(10, 7), "A ten-frame with 7 dots has 3 empty boxes, so 7 and 3 make 10."),
        (s.bond(5, 2), "Hold up 5 fingers and fold down 2. 3 are still up, so 2 and 3 make 5."),
        (s.subitize(7, "ten_frame"), "A full row is 5, and 2 more makes 7."),
        (s.subitize(8, "dice"), "Two dice: 4 and 4 make 8."),
        (s.number_line(100, 37, 10), "The arrow is a little past halfway from 30 to 40, so it points to about 37."),
        (s.skip_count(5, 15), "Each number is 5 more than the one before: 15 + 5 = 20."),
    ],
)
def test_hints_name_a_concrete_strategy(hint, expected):
    assert hint == expected


def test_a_times_hint_only_leans_on_tables_learned_earlier():
    # 6 × 9: the 9s are learned before the 6s, so the hint uses "×10 then take one away".
    assert Mul(6, 9).hint() == "6 × 9: 6 × 10 is 60, take away one 6 to get 54."
    assert Mul(3, 5).hint() == "3 × 5: half of 3 × 10. Half of 30 is 15."
    assert Mul(1, 8).hint() == "1 × 8: times 1 keeps a number the same, so 8."


def test_the_make_ten_skill_teaches_make_ten_even_for_near_doubles():
    assert Add(8, 9).hint() == "8 + 9: double 8 is 16, and 1 more is 17."
    problem = build_problem(SKILLS["make_ten"], Add(9, 8), rng=random.Random(0), show_picture=False, problem_id="p")
    assert "make 10" in problem.hint


def test_every_hint_is_short_kind_and_reaches_the_answer():
    unkind = re.compile(r"\b(wrong|mistake|incorrect|fail|bad|careless|should have|easy)\b", re.IGNORECASE)
    for skill in LADDER:
        for fact in skill.facts:
            hint = fact.hint(skill.hint_style)
            assert 10 <= len(hint) <= 120, hint
            assert not unkind.search(hint), hint
            assert re.search(rf"(?<!\d){fact.answer}(?!\d)", hint), f"{fact.key}: {hint}"


def test_hint_grammar_for_ones():
    assert Bond(10, 9).hint() == "A ten-frame with 9 dots has 1 empty box, so 9 and 1 make 10."
    assert Bond(5, 4).hint() == "Hold up 5 fingers and fold down 4. 1 is still up, so 4 and 1 make 5."
    assert Sub(10, 9).hint() == f"10 {M} 9: 9 and 1 more make 10, so 1."
    assert Add(1, 10).hint() == "1 + 10: 1 ten and 1 one make 11."
    assert Subitize(1, "random").hint() == "There is just 1 dot."
    assert NumberLinePoint(10, 6).hint() == "Follow the arrow down to the line and read the number there: 6."
