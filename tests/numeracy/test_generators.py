"""Every skill's problems are valid at many seeds: right answers, honest pictures, no negatives."""

import random

import pytest

from legible.numeracy import LADDER, SKILLS, build_problem
from legible.numeracy.facts import Compare, Div, Mul, NumberLinePoint, Sub
from legible.numeracy.visuals import dice_faces

from .support import assert_picture_matches, nice_ticks, solve

SEEDS = range(12)
CONTRACT_KEYS = {"problem_id", "kind", "prompt", "visual", "answer_format", "choices"}
KINDS = {"subitize", "compare", "bond", "add", "sub", "mul", "div", "number_line"}


def problems_for(skill, *, show_picture):
    for seed in SEEDS:
        rng = random.Random(seed)
        for fact in skill.facts:
            yield build_problem(skill, fact, rng=rng, show_picture=show_picture, problem_id=f"p{seed}")


@pytest.mark.parametrize("skill", LADDER, ids=lambda skill: skill.id)
def test_every_problem_can_be_solved_from_what_the_learner_sees(skill):
    for show_picture in (True, False):
        for problem in problems_for(skill, show_picture=show_picture):
            public = problem.public()
            assert set(public) == CONTRACT_KEYS and public["kind"] in KINDS
            assert 0 < len(public["prompt"]) <= 40
            answer = problem.fact.answer
            assert solve(public) == answer, public
            if public["answer_format"] == "integer":
                assert isinstance(answer, int) and answer >= 0
                assert public["choices"] is None
            else:
                choices = public["choices"]
                assert answer in choices and len(set(choices)) == len(choices) and 2 <= len(choices) <= 9
                assert all(choice.isdigit() for choice in choices)


@pytest.mark.parametrize("skill", LADDER, ids=lambda skill: skill.id)
def test_pictures_match_their_problem_and_the_skill(skill):
    for problem in problems_for(skill, show_picture=True):
        # Every fact has at least one picture its skill teaches with.
        assert problem.visual is not None, f"{skill.id} has no picture for {problem.fact.key}"
        assert problem.visual["type"] in skill.visuals
        assert_picture_matches(problem.fact, problem.visual)


@pytest.mark.parametrize("skill", LADDER, ids=lambda skill: skill.id)
def test_without_picture_support_only_essential_pictures_remain(skill):
    for problem in problems_for(skill, show_picture=False):
        assert (problem.visual is not None) == problem.fact.needs_picture


def test_counting_and_number_line_problems_always_show_their_picture():
    for skill_id in ("subitize_5", "subitize_10", "number_line_10", "number_line_100"):
        assert all(fact.needs_picture for fact in SKILLS[skill_id].facts)


def test_number_line_targets():
    near = SKILLS["number_line_10"].facts
    assert sorted(f.target for f in near) == list(range(1, 10)) and {f.tolerance for f in near} == {0}
    far = SKILLS["number_line_100"].facts
    # Only the tens are labelled on a 0-100 line, so every target is an estimate.
    assert all(f.target % 10 and 0 < f.target < 100 for f in far) and len(far) == 90
    assert {f.tolerance for f in far} == {3}
    assert all(isinstance(f, NumberLinePoint) for f in (*near, *far))


def test_turn_around_facts_share_a_memory_key_but_vary_how_they_are_shown():
    shown = {Mul(7, 8).presented(random.Random(seed)).prompt() for seed in range(30)}
    assert shown == {"7 × 8", "8 × 7"}
    assert Mul(7, 8).key == Mul(8, 7).key == "mul:7x8"
    assert Compare(4, 7).key == Compare(7, 4).key == "compare:4:7"


def test_facts_reject_impossible_numbers():
    with pytest.raises(ValueError):
        Sub(3, 5)  # would be negative
    with pytest.raises(ValueError):
        Div(7, 2)  # not a whole number
    with pytest.raises(ValueError):
        Compare(4, 4)


def test_ports_of_the_web_layouts():
    assert dice_faces(0) == [] and dice_faces(5) == [5] and dice_faces(6) == [6]
    assert dice_faces(7) == [4, 3] and dice_faces(12) == [6, 6] and dice_faces(13) == [5, 5, 3]
    assert nice_ticks(0, 10) == list(range(11))
    assert nice_ticks(0, 100) == list(range(0, 101, 10))
    assert 0.5 in nice_ticks(0, 5)  # why lines never run to 5
