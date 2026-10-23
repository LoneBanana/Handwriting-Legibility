from legible.numeracy import LADDER, LEVELS_PER_SKILL, SKILLS, round_settings, skill_for_level, skill_index
from legible.numeracy.rounds import review_count
from legible.progression import TRACKS

TRACK = TRACKS["numeracy.fluency"]


def test_about_25_skills_with_unique_ids_and_short_labels():
    assert 24 <= len(LADDER) <= 26
    ids = [skill.id for skill in LADDER]
    assert len(set(ids)) == len(ids)
    assert all(0 < len(skill.label) <= 40 for skill in LADDER)


def test_level_picks_the_skill_in_steps_of_twelve():
    for level in range(0, 400):
        assert skill_index(level) == min(level // 12, len(LADDER) - 1)
    assert skill_for_level(0).id == "subitize_5"
    assert skill_for_level(11).id == "subitize_5"
    assert skill_for_level(12).id == "subitize_10"
    assert skill_for_level(TRACK.top_level).id == "mixed"


def test_every_skill_is_reachable_on_the_track():
    assert (len(LADDER) - 1) * LEVELS_PER_SKILL <= TRACK.top_level


def test_harder_skills_get_more_time_and_factors_never_decrease():
    # Never decreasing means review of an earlier skill never gets less time than that skill had.
    factors = [skill.time_factor for skill in LADDER]
    assert factors == sorted(factors)
    assert factors[0] < 1 < factors[-1]


def test_number_sense_comes_before_facts_and_tables_follow_the_strategy_order():
    order = [skill.id for skill in LADDER]
    position = order.index
    assert position("subitize_5") < position("subitize_10") < position("compare_10") < position("number_line_10")
    assert position("number_line_10") < position("bonds_5") < position("bonds_10") < position("add_sub_5")
    assert position("bonds_10") < position("make_ten") and position("doubles") < position("near_doubles")
    assert position("add_20") < position("sub_20") < position("skip_count") < position("mul_2")
    tables = [skill_id for skill_id in order if skill_id.startswith("mul_")]
    assert tables == ["mul_2", "mul_10", "mul_5", "mul_3", "mul_4", "mul_9", "mul_6", "mul_7", "mul_8"]
    assert order[-2:] == ["div", "mixed"]


def test_fact_keys_are_unique_within_each_skill():
    for skill in LADDER:
        keys = [fact.key for fact in skill.facts]
        assert len(keys) == len(set(keys)), skill.id


def test_each_skill_has_enough_facts_for_its_share_of_a_round():
    for index, skill in enumerate(LADDER):
        last_level = min(index * LEVELS_PER_SKILL + LEVELS_PER_SKILL - 1, TRACK.top_level)
        settings = round_settings(last_level)
        assert len(skill.facts) >= settings.problems - review_count(settings.problems, index > 0), skill.id


def test_mixed_practice_covers_every_fluency_fact():
    mixed = {fact.key for fact in SKILLS["mixed"].facts}
    for skill_id in ("add_20", "sub_20", "div", *(f"mul_{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10))):
        assert {fact.key for fact in SKILLS[skill_id].facts} <= mixed


def test_times_tables_run_from_1_to_10_and_division_is_their_inverse():
    for n in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        assert sorted(fact.answer for fact in SKILLS[f"mul_{n}"].facts) == [n * b for b in range(1, 11)]
    divisions = {(fact.dividend, fact.divisor) for fact in SKILLS["div"].facts}
    assert divisions == {(d * q, d) for d in range(2, 11) for q in range(1, 11)}
