"""The pictures a math problem can show, exactly as the web app draws them.

Visuals are JSON-ready dicts matching the ``Visual`` type in docs/ARCHITECTURE.md. How
``web/src/features/math`` draws each one limits what the generator may ask for:

* ``dots``: ``dice`` shows one die face up to 6 and two near-equal faces up to 12
  (7 is drawn as 4 and 3), so two dice can picture the two halves of a double or near
  double. ``random`` scatters the dots and ``line`` puts them in one evenly spaced row.
* ``ten_frame``: rows of five boxes filled in reading order. ``frames`` is how many
  frames are drawn; a second one appears by itself once ``filled`` passes 10.
* ``array``: ``rows`` × ``cols`` dots. The web app clips arrays at 15 × 15.
* ``number_line``: ticks are spaced by a "nice" step of about a tenth of the range and
  every tick is labelled. That is why lines only run from 0 to 10, 20, 50 or 100: their
  ticks are whole numbers (1s, 2s, 5s or 10s) with no crowded last tick.
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union

DotsLayout = Literal["dice", "random", "line"]
VisualType = Literal["dots", "ten_frame", "array", "number_line"]

# Line lengths whose ticks come out as whole numbers, mapped to the gap between labels.
NUMBER_LINE_STEPS: dict[int, int] = {10: 1, 20: 2, 50: 5, 100: 10}


class DotsVisual(TypedDict):
    type: Literal["dots"]
    count: int
    layout: DotsLayout


class TenFrameVisual(TypedDict):
    type: Literal["ten_frame"]
    filled: int
    frames: int


class ArrayVisual(TypedDict):
    type: Literal["array"]
    rows: int
    cols: int


class NumberLineVisual(TypedDict):
    type: Literal["number_line"]
    min: int
    max: int
    target: int | None


Visual = Union[DotsVisual, TenFrameVisual, ArrayVisual, NumberLineVisual]


def dots(count: int, layout: DotsLayout) -> DotsVisual:
    return {"type": "dots", "count": count, "layout": layout}


def ten_frame(filled: int, frames: int = 1) -> TenFrameVisual:
    return {"type": "ten_frame", "filled": filled, "frames": frames}


def array(rows: int, cols: int) -> ArrayVisual:
    return {"type": "array", "rows": rows, "cols": cols}


def number_line(end: int, target: int | None = None) -> NumberLineVisual:
    """A line from 0 to ``end`` (one of :data:`NUMBER_LINE_STEPS`), with an optional arrow."""
    if end not in NUMBER_LINE_STEPS:
        raise ValueError(f"number lines run to 10, 20, 50 or 100, not {end}")
    return {"type": "number_line", "min": 0, "max": end, "target": target}


def line_end_for(value: int) -> int:
    """The shortest drawable number line (starting at 0) that reaches ``value``."""
    for end in sorted(NUMBER_LINE_STEPS):
        if value <= end:
            return end
    raise ValueError(f"no number line reaches {value}")


def dice_faces(count: int) -> list[int]:
    """How the web app splits ``count`` dots into die faces (mirrors ``diceGroups``)."""
    if count <= 0:
        return []
    if count <= 6:
        return [count]
    if count <= 12:
        return [(count + 1) // 2, count // 2]
    faces = []
    while count > 0:
        faces.append(min(5, count))
        count -= faces[-1]
    return faces
