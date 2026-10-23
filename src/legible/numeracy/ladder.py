"""The skill ladder: 26 steps from seeing small amounts to fluent times tables.

The order follows what dyscalculia research says about how number facts are built:

* **Number sense before facts.** Seeing small amounts at a glance (subitizing),
  comparing numbers and placing them on a number line are core weaknesses in
  dyscalculia and predict later arithmetic, so they come first.
* **Concrete → pictorial → abstract.** Skills teach with pictures (dice, ten-frames,
  arrays, number lines). The track's ``visual_support`` fades them from 100% to 0% over
  the year, so facts move from "see it" to "know it".
* **Strategies, not counting by ones.** Facts are grouped by the strategy that solves
  them: bonds to 10 before make-ten, doubles before near doubles, then times tables in
  an order where each one leans on tables already known: ×2, ×10, ×5 first; then ×3
  (double + one group), ×4 (double double), ×9 (×10 − one group); then ×6 (×5 + one
  group), ×7 (×5 + ×2) and ×8 (double ×4). Division comes last, as times facts in
  reverse.
* **Retrieval and maintenance.** The top level is mixed practice of every fluency fact.
  Every round also mixes in review of earlier skills (see rounds.py).

Each skill spans :data:`LEVELS_PER_SKILL` levels of the ``numeracy.fluency`` track (26
skills × 12 levels covers its 300 levels), so a learner has at least 12 perfect days on
each skill before moving on. Harder skills get proportionally more time per problem
(``time_factor`` multiplies the track's time limit). The factors never go down along
the ladder, so review problems from an earlier skill never get less time than that
skill gave them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .facts import Add, Bond, Compare, Div, Fact, Mul, NumberLinePoint, SkipCount, Sub, Subitize
from .visuals import VisualType

LEVELS_PER_SKILL = 12
TIMES_TABLE_ORDER = (2, 10, 5, 3, 4, 9, 6, 7, 8)


@dataclass(frozen=True)
class Skill:
    id: str
    label: str                       # learner-facing, shown as the round's title
    time_factor: float               # × the track's time limit: harder skills get more time
    facts: tuple[Fact, ...]          # everything the skill practises, one entry per fact key
    visuals: tuple[VisualType, ...]  # the kinds of picture this skill teaches with
    hint_style: str | None = None    # e.g. "make_ten": hints use this skill's strategy when it applies


# ------------------------------------------------------------------------ fact sets


def _subitizing_to_5() -> tuple[Fact, ...]:
    return tuple(Subitize(n, layout) for layout in ("dice", "random", "line") for n in range(1, 6))


def _subitizing_to_10() -> tuple[Fact, ...]:
    # Ten-frames show 6-10 as "5 and some more"; 7-10 as dice are two faces (4 and 3, ...).
    frames = (Subitize(n, "ten_frame") for n in range(1, 11))
    dice = (Subitize(n, "dice") for n in range(6, 11))
    return (*frames, *dice)


def _comparisons(top: int) -> tuple[Fact, ...]:
    return tuple(Compare(a, b) for a in range(1, top + 1) for b in range(a + 1, top + 1))


def _number_line(end: int, within: int, skip_every: int | None = None) -> tuple[Fact, ...]:
    """Targets 1 .. end-1; ``skip_every`` leaves out labelled ticks so every target is an estimate."""
    return tuple(
        NumberLinePoint(end, target, within)
        for target in range(1, end)
        if skip_every is None or target % skip_every
    )


def _bonds(whole: int) -> tuple[Fact, ...]:
    return tuple(Bond(whole, part, missing_first) for missing_first in (False, True) for part in range(whole + 1))


def _pairs(low: int, high: int) -> list[tuple[int, int]]:
    """Every a <= b (both 1-10) with a total from ``low`` to ``high``."""
    return [(a, b) for b in range(1, 11) for a in range(1, b + 1) if low <= a + b <= high]


def _sums(low: int, high: int) -> tuple[Fact, ...]:
    return tuple(Add(a, b) for a, b in _pairs(low, high))


def _take_aways(low: int, high: int) -> tuple[Fact, ...]:
    """Every whole - part (part at least 1, answer at least 1) for wholes from ``low`` to ``high``."""
    return tuple(Sub(whole, part) for whole in range(low, high + 1) for part in range(1, whole))


def _doubles() -> tuple[Fact, ...]:
    return tuple(Add(n, n) for n in range(1, 11))


def _near_doubles() -> tuple[Fact, ...]:
    return tuple(Add(n, n + 1) for n in range(1, 10))


def _make_ten() -> tuple[Fact, ...]:
    nines = (Add(9, b) for b in range(2, 10))
    eights = (Add(8, b) for b in range(3, 9))  # 8 + 9 is already among the nines
    return (*nines, *eights)


def _teen_differences() -> tuple[Fact, ...]:
    """The take-away partners of every sum from 11 to 20, e.g. 13 - 5 and 13 - 8."""
    facts: dict[str, Fact] = {}
    for a, b in _pairs(11, 20):
        for part in (a, b):
            difference = Sub(a + b, part)
            facts.setdefault(difference.key, difference)
    return tuple(facts.values())


def _skip_counts() -> tuple[Fact, ...]:
    return tuple(SkipCount(step, k * step) for step in (2, 5, 10) for k in range(8))


def _times_table(n: int) -> tuple[Fact, ...]:
    return tuple(Mul(n, b) for b in range(1, 11))


def _divisions() -> tuple[Fact, ...]:
    return tuple(Div(divisor * quotient, divisor) for divisor in range(2, 11) for quotient in range(1, 11))


def _mixed(*skills: Skill) -> tuple[Fact, ...]:
    facts: dict[str, Fact] = {}
    for skill in skills:
        for fact in skill.facts:
            facts.setdefault(fact.key, fact)
    return tuple(facts.values())


# --------------------------------------------------------------------------- ladder

_ADD_20 = Skill("add_20", "Adding within 20", 1.2, _sums(11, 20), ("ten_frame", "number_line"))
_SUB_20 = Skill("sub_20", "Taking away within 20", 1.3, _teen_differences(), ("ten_frame", "number_line"))
_TABLES = tuple(
    Skill(
        f"mul_{n}",
        f"The {n} times table",
        # ×2, ×10, ×5 lean on doubling and counting in tens; ×3, ×4, ×9 take one derived
        # step from them; ×6, ×7, ×8 combine two.
        1.3 if n in (2, 10, 5) else 1.4 if n in (3, 4, 9) else 1.5,
        _times_table(n),
        ("array",),
    )
    for n in TIMES_TABLE_ORDER
)
_DIVISION = Skill("div", "Dividing with times tables", 1.5, _divisions(), ("array",))

LADDER: tuple[Skill, ...] = (
    # Number sense: see amounts without counting, compare them, place them on a line.
    Skill("subitize_5", "Seeing amounts up to 5", 0.6, _subitizing_to_5(), ("dots",)),
    Skill("subitize_10", "Seeing amounts up to 10", 0.7, _subitizing_to_10(), ("ten_frame", "dots")),
    Skill("compare_10", "Which number is more?", 0.7, _comparisons(10), ("number_line",)),
    # Every whole number on a 0-10 line has a labelled tick, so the answer must be exact.
    Skill("number_line_10", "Number line to 10", 0.8, _number_line(10, within=0), ("number_line",)),
    # Part-whole thinking: the base of make-ten and of taking away.
    Skill("bonds_5", "Pairs that make 5", 0.9, _bonds(5), ("dots",)),
    Skill("bonds_10", "Pairs that make 10", 0.9, _bonds(10), ("ten_frame",)),
    # Adding and taking away together, as fact families (2 + 3 = 5, 5 − 3 = 2).
    Skill(
        "add_sub_5",
        "Adding and taking away within 5",
        1.0,
        (*_sums(2, 5), *_take_aways(2, 5)),
        ("ten_frame", "number_line"),
    ),
    Skill(
        "add_sub_10",
        "Adding and taking away within 10",
        1.0,
        (*_sums(6, 10), *_take_aways(6, 10)),
        ("ten_frame", "number_line"),
    ),
    # Strategies for the facts to 20.
    Skill("doubles", "Doubles", 1.0, _doubles(), ("array", "dots")),
    Skill("near_doubles", "Near doubles", 1.1, _near_doubles(), ("dots", "ten_frame")),
    Skill("make_ten", "Making 10 to add 9 and 8", 1.2, _make_ten(), ("ten_frame",), hint_style="make_ten"),
    _ADD_20,
    _SUB_20,
    # Tens are labelled, so an estimate within 3 of the arrow counts.
    Skill("number_line_100", "Number line to 100", 1.3, _number_line(100, within=3, skip_every=10), ("number_line",)),
    Skill("skip_count", "Counting in 2s, 5s and 10s", 1.3, _skip_counts(), ("number_line",)),
    *_TABLES,
    _DIVISION,
    # The top level: maintenance practice of every fluency fact.
    Skill(
        "mixed",
        "Mixed facts",
        1.5,
        _mixed(_ADD_20, _SUB_20, *_TABLES, _DIVISION),
        ("ten_frame", "number_line", "array"),
    ),
)

SKILLS: dict[str, Skill] = {skill.id: skill for skill in LADDER}


def skill_index(level: int) -> int:
    return min(max(level, 0) // LEVELS_PER_SKILL, len(LADDER) - 1)


def skill_for_level(level: int) -> Skill:
    return LADDER[skill_index(level)]
