"""Independent checks used by the numeracy tests: they know only what the learner sees."""

from __future__ import annotations

import math
import re
from typing import Any

from legible.numeracy.facts import Add, Bond, Compare, Div, Fact, Mul, NumberLinePoint, SkipCount, Sub, Subitize

_BINARY = re.compile(r"(\d+) ([+−×÷]) (\d+)")
_BOND = re.compile(r"(\d+|\?) and (\d+|\?) make (\d+)")
_SKIP = re.compile(r"(\d+), (\d+), (\d+), \?")


def solve(public: dict[str, Any]) -> int | str:
    """Works out a problem's answer from its prompt, picture and choices alone."""
    prompt, visual = public["prompt"], public["visual"]
    if public["kind"] == "subitize":
        return visual["count"] if visual["type"] == "dots" else visual["filled"]
    if public["kind"] == "number_line":
        return visual["target"]
    if public["kind"] == "compare":
        return max(public["choices"], key=int)
    if match := _BOND.fullmatch(prompt):
        known = match[1] if match[1] != "?" else match[2]
        return int(match[3]) - int(known)
    if match := _SKIP.fullmatch(prompt):
        a, b, c = (int(match[i]) for i in (1, 2, 3))
        assert b - a == c - b, prompt
        return c + (c - b)
    match = _BINARY.fullmatch(prompt)
    assert match, f"unexpected prompt {prompt!r}"
    a, op, b = int(match[1]), match[2], int(match[3])
    if op == "+":
        return a + b
    if op == "−":
        return a - b
    if op == "×":
        return a * b
    assert a % b == 0, prompt
    return a // b


def nice_ticks(low: float, high: float, approx: int = 10) -> list[float]:
    """Port of ``niceTicks`` in web/src/features/math/layouts.ts."""
    if not high > low:
        return [low]
    raw = (high - low) / approx
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        step = 1.0
    elif normalized <= 2:
        step = 2.0
    elif normalized <= 2.5:
        step = 2.5
    elif normalized <= 5:
        step = 5.0
    else:
        step = 10.0
    step *= magnitude
    ticks = []
    value = math.ceil(low / step) * step
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    if ticks[0] != low:
        ticks.insert(0, low)
    if ticks[-1] != high:
        ticks.append(high)
    return ticks


def assert_picture_matches(fact: Fact, visual: dict[str, Any]) -> None:
    """Every picture must show the amounts of its own problem (see the rules in facts.py)."""
    kind = visual["type"]
    if kind == "number_line":
        ticks = nice_ticks(visual["min"], visual["max"])
        gaps = {b - a for a, b in zip(ticks, ticks[1:])}
        assert all(float(t).is_integer() for t in ticks) and len(gaps) == 1, (fact, ticks)
        assert visual["min"] == 0
    if kind == "array":
        assert 1 <= visual["rows"] <= 15 and 1 <= visual["cols"] <= 15

    if isinstance(fact, Subitize):
        if fact.layout == "ten_frame":
            assert visual == {"type": "ten_frame", "filled": fact.count, "frames": 1}
        else:
            assert visual == {"type": "dots", "count": fact.count, "layout": fact.layout}
            if fact.layout == "dice":
                assert fact.count <= 12
    elif isinstance(fact, Compare):
        assert kind == "number_line" and visual["target"] is None
        assert max(fact.a, fact.b) <= visual["max"]
    elif isinstance(fact, NumberLinePoint):
        assert visual == {"type": "number_line", "min": 0, "max": fact.end, "target": fact.answer}
    elif isinstance(fact, Bond):
        if kind == "dots":
            assert visual == {"type": "dots", "count": fact.whole, "layout": "line"}
        else:
            assert visual == {"type": "ten_frame", "filled": fact.part, "frames": 1} and fact.whole == 10
    elif isinstance(fact, SkipCount):
        assert kind == "number_line" and visual["target"] is None
        step = nice_ticks(0, visual["max"])[1]
        assert step == fact.step and fact.answer <= visual["max"]
    elif isinstance(fact, Add):
        if kind == "array":
            assert fact.a == fact.b and visual["rows"] == 2 and visual["rows"] * visual["cols"] == fact.answer
        elif kind == "dots":
            # Two dice, one per number: the web app draws 7 to 12 dots as two near-equal faces.
            assert visual["layout"] == "dice" and visual["count"] == fact.answer
            assert 7 <= fact.answer <= 12 and abs(fact.a - fact.b) <= 1
        elif kind == "ten_frame":
            assert visual["filled"] == max(fact.a, fact.b)
            assert visual["frames"] == (1 if fact.answer <= 10 else 2)
        else:
            assert kind == "number_line" and visual["target"] is None and fact.answer <= visual["max"]
    elif isinstance(fact, Sub):
        if kind == "ten_frame":
            assert visual["filled"] == fact.a and visual["frames"] * 10 >= fact.a
        else:
            assert kind == "number_line" and visual["target"] is None and fact.a <= visual["max"]
    elif isinstance(fact, Mul):
        assert kind == "array" and (visual["rows"], visual["cols"]) == (fact.a, fact.b)
    elif isinstance(fact, Div):
        assert kind == "array" and visual["rows"] == fact.divisor and visual["cols"] == fact.answer
        assert visual["rows"] * visual["cols"] == fact.dividend
    else:
        raise AssertionError(f"no picture rule for {fact!r}")
