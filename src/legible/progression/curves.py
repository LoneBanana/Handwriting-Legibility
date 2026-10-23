"""Parameter curves: how a track's difficulty settings move with its level.

A curve maps a level to a value between ``start`` (level 0) and ``target`` (level
``levels_to_target`` and beyond). Geometric curves change by the same *ratio* at every
level, which keeps each step equally (and barely) noticeable: perceived differences
in speed or duration scale with relative change (Weber's law), so a 0.2% step is as
gentle at 170 words per minute as it was at 110. Linear curves are for values that
pass through zero or count discrete things (choices, hints, percentages of support).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Curve:
    start: float
    target: float
    unit: str = ""
    label: str = ""
    shape: Literal["geometric", "linear"] = "geometric"
    integer: bool = False

    def fraction(self, level: int, levels_to_target: int) -> float:
        """How far along the curve ``level`` is, in [0, 1]."""
        if levels_to_target <= 0:
            return 1.0
        return min(max(level, 0), levels_to_target) / levels_to_target

    def at(self, level: int, levels_to_target: int) -> float:
        t = self.fraction(level, levels_to_target)
        if self.shape == "geometric" and self.start > 0 and self.target > 0:
            value = self.start * (self.target / self.start) ** t
        else:
            value = self.start + (self.target - self.start) * t
        if self.integer:
            return float(math.floor(value + 0.5))
        return value

    def step_ratio(self, levels_to_target: int) -> float:
        """Relative change per level of a geometric curve (e.g. 1.0016 = +0.16%)."""
        if self.shape != "geometric" or self.start <= 0 or self.target <= 0 or levels_to_target <= 0:
            return 1.0
        return (self.target / self.start) ** (1.0 / levels_to_target)
