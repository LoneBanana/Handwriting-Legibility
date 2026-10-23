"""Math facts: the unit of practice, memory and feedback.

A fact (``7 × 8``, ``3 and ? make 5``, "how many dots in this dice pattern") knows its
answer, the prompt the learner sees, every picture that is consistent with it, and the
strategy hint for when it was answered wrongly or slowly.

Each fact has a ``key`` under which the learner's accuracy and speed are remembered
(``FactMemory.fact``). Turn-around facts share a key ("mul:7x8" covers 7 × 8 and
8 × 7): knowing one gives the other for free, which halves what has to be learned.
The order shown is picked per problem (:meth:`Fact.presented`).

Pictures follow one rule per kind, so a picture never contradicts its problem:

* counting and number-line problems show exactly the amount or position asked about;
* adding shows the bigger number in a ten-frame (count on from it, or fill the frame to
  make ten), the two halves of a double as two rows or two dice, or a number line;
* taking away shows the starting amount (the learner takes away from it);
* bonds show the whole as a row of dots (to 5) or the known part in a ten-frame (to 10),
  where the empty boxes are the missing part;
* times and division facts show an array: ``a × b`` is ``a`` rows of ``b``, and
  ``a ÷ b`` is ``a`` dots shared into ``b`` rows.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar, Literal

from . import strategies
from .strategies import MINUS
from .visuals import NUMBER_LINE_STEPS, Visual, array, dice_faces, dots, line_end_for, number_line, ten_frame

Kind = Literal["subitize", "compare", "bond", "add", "sub", "mul", "div", "number_line"]
AnswerFormat = Literal["integer", "choice"]


class Fact(ABC):
    kind: ClassVar[Kind]
    answer_format: ClassVar[AnswerFormat] = "integer"
    # True when the question means nothing without its picture ("How many dots?").
    needs_picture: ClassVar[bool] = False

    @property
    @abstractmethod
    def key(self) -> str:
        """Identity in the learner's fact memory, e.g. ``"mul:7x8"``."""

    @property
    @abstractmethod
    def answer(self) -> int | str:
        """An int, or for choice problems the correct choice."""

    @property
    def tolerance(self) -> int:
        """How far an answer may be from :attr:`answer` and still count (number-line estimates)."""
        return 0

    @abstractmethod
    def prompt(self) -> str: ...

    @abstractmethod
    def hint(self, style: str | None = None) -> str:
        """The strategy shown after a wrong or slow answer (``style`` lets a skill pick its own)."""

    def pictures(self) -> list[Visual]:
        """Every picture consistent with this fact; each skill chooses among the kinds it teaches with."""
        return []

    def choices(self) -> list[str] | None:
        return None

    def presented(self, rng: random.Random) -> Fact:
        """The version shown to the learner: turn-around facts swap their numbers half the time."""
        return self


def _swap_half_the_time(fact: Fact, rng: random.Random, **swapped: int) -> Fact:
    return replace(fact, **swapped) if rng.random() < 0.5 else fact  # type: ignore[type-var]


# ---------------------------------------------------------------------- number sense


@dataclass(frozen=True)
class Subitize(Fact):
    """How many dots: a dice pattern, a scatter, a row, or a ten-frame."""

    count: int
    layout: Literal["dice", "random", "line", "ten_frame"]

    kind = "subitize"
    needs_picture = True

    @property
    def key(self) -> str:
        return f"subitize:{self.layout}:{self.count}"

    @property
    def answer(self) -> int:
        return self.count

    def prompt(self) -> str:
        return "How many dots?"

    def pictures(self) -> list[Visual]:
        if self.layout == "ten_frame":
            return [ten_frame(self.count)]
        return [dots(self.count, self.layout)]

    def hint(self, style: str | None = None) -> str:
        return strategies.subitize(self.count, self.layout)


@dataclass(frozen=True)
class Compare(Fact):
    """Which of two different numbers is more (comparing written numbers)."""

    a: int
    b: int

    kind = "compare"
    answer_format = "choice"

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("compare needs two different numbers")

    @property
    def key(self) -> str:
        return f"compare:{min(self.a, self.b)}:{max(self.a, self.b)}"

    @property
    def answer(self) -> str:
        return str(max(self.a, self.b))

    def prompt(self) -> str:
        return f"Which is more, {self.a} or {self.b}?"

    def choices(self) -> list[str]:
        return [str(self.a), str(self.b)]

    def pictures(self) -> list[Visual]:
        return [number_line(line_end_for(max(self.a, self.b)))]

    def presented(self, rng: random.Random) -> Fact:
        return _swap_half_the_time(self, rng, a=self.b, b=self.a)

    def hint(self, style: str | None = None) -> str:
        return strategies.compare(self.a, self.b)


@dataclass(frozen=True)
class NumberLinePoint(Fact):
    """What number an arrow on a 0-to-``end`` line points at; estimates within ``within`` count."""

    end: int
    target: int
    within: int = 0

    kind = "number_line"
    needs_picture = True

    @property
    def key(self) -> str:
        return f"line:{self.end}:{self.target}"

    @property
    def answer(self) -> int:
        return self.target

    @property
    def tolerance(self) -> int:
        return self.within

    def prompt(self) -> str:
        return "What number is at the arrow?"

    def pictures(self) -> list[Visual]:
        return [number_line(self.end, self.target)]

    def hint(self, style: str | None = None) -> str:
        return strategies.number_line(self.end, self.target, NUMBER_LINE_STEPS[self.end])


@dataclass(frozen=True)
class Bond(Fact):
    """Part-whole: the number that goes with ``part`` to make ``whole``."""

    whole: int
    part: int
    missing_first: bool = False  # "? and 3 make 5" rather than "3 and ? make 5"

    kind = "bond"

    @property
    def key(self) -> str:
        parts = f"?+{self.part}" if self.missing_first else f"{self.part}+?"
        return f"bond:{self.whole}:{parts}"

    @property
    def answer(self) -> int:
        return self.whole - self.part

    def prompt(self) -> str:
        if self.missing_first:
            return f"? and {self.part} make {self.whole}"
        return f"{self.part} and ? make {self.whole}"

    def pictures(self) -> list[Visual]:
        if self.whole <= 5:
            return [dots(self.whole, "line")]
        if self.whole == 10:
            return [ten_frame(self.part)]
        return []

    def hint(self, style: str | None = None) -> str:
        return strategies.bond(self.whole, self.part)


# ------------------------------------------------------------- adding, taking away


@dataclass(frozen=True)
class Add(Fact):
    a: int
    b: int

    kind = "add"

    @property
    def key(self) -> str:
        return f"add:{min(self.a, self.b)}+{max(self.a, self.b)}"

    @property
    def answer(self) -> int:
        return self.a + self.b

    def prompt(self) -> str:
        return f"{self.a} + {self.b}"

    def pictures(self) -> list[Visual]:
        total, big = self.a + self.b, max(self.a, self.b)
        pictures: list[Visual] = []
        if self.a == self.b and self.a <= 15:
            pictures.append(array(2, self.a))
        if total >= 7 and sorted(dice_faces(total)) == sorted((self.a, self.b)):
            pictures.append(dots(total, "dice"))  # two dice, one per number
        if total <= 20:
            pictures.append(ten_frame(big, 1 if total <= 10 else 2))
            pictures.append(number_line(line_end_for(total)))
        return pictures

    def presented(self, rng: random.Random) -> Fact:
        return _swap_half_the_time(self, rng, a=self.b, b=self.a)

    def hint(self, style: str | None = None) -> str:
        return strategies.add(self.a, self.b, prefer_make_ten=style == "make_ten")


@dataclass(frozen=True)
class Sub(Fact):
    a: int
    b: int

    kind = "sub"

    def __post_init__(self) -> None:
        if self.b > self.a:
            raise ValueError("take-away facts never go below zero")

    @property
    def key(self) -> str:
        return f"sub:{self.a}-{self.b}"

    @property
    def answer(self) -> int:
        return self.a - self.b

    def prompt(self) -> str:
        return f"{self.a} {MINUS} {self.b}"

    def pictures(self) -> list[Visual]:
        if self.a > 20:
            return []
        return [ten_frame(self.a, 1 if self.a <= 10 else 2), number_line(line_end_for(self.a))]

    def hint(self, style: str | None = None) -> str:
        return strategies.subtract(self.a, self.b)


@dataclass(frozen=True)
class SkipCount(Fact):
    """The next number when counting in ``step``s, e.g. "5, 10, 15, ?"."""

    step: int
    start: int  # the first number shown, a multiple of ``step``

    kind = "add"

    @property
    def key(self) -> str:
        return f"skip:{self.step}:{self.start}"

    @property
    def terms(self) -> tuple[int, int, int]:
        return self.start, self.start + self.step, self.start + 2 * self.step

    @property
    def answer(self) -> int:
        return self.start + 3 * self.step

    def prompt(self) -> str:
        return ", ".join(str(term) for term in self.terms) + ", ?"

    def pictures(self) -> list[Visual]:
        # A line to 10 steps has a labelled tick on every number of the count.
        end = 10 * self.step
        return [number_line(end)] if end in NUMBER_LINE_STEPS and self.answer <= end else []

    def hint(self, style: str | None = None) -> str:
        return strategies.skip_count(self.step, self.terms[-1])


# ------------------------------------------------------------- times and dividing


@dataclass(frozen=True)
class Mul(Fact):
    a: int
    b: int

    kind = "mul"

    @property
    def key(self) -> str:
        return f"mul:{min(self.a, self.b)}x{max(self.a, self.b)}"

    @property
    def answer(self) -> int:
        return self.a * self.b

    def prompt(self) -> str:
        return f"{self.a} × {self.b}"

    def pictures(self) -> list[Visual]:
        return [array(self.a, self.b)] if max(self.a, self.b) <= 15 else []

    def presented(self, rng: random.Random) -> Fact:
        return _swap_half_the_time(self, rng, a=self.b, b=self.a)

    def hint(self, style: str | None = None) -> str:
        return strategies.multiply(self.a, self.b)


@dataclass(frozen=True)
class Div(Fact):
    """Division as the inverse of a times fact: ``dividend ÷ divisor``, always exact."""

    dividend: int
    divisor: int

    kind = "div"

    def __post_init__(self) -> None:
        if self.divisor <= 0 or self.dividend % self.divisor:
            raise ValueError("division facts divide exactly")

    @property
    def key(self) -> str:
        return f"div:{self.dividend}/{self.divisor}"

    @property
    def answer(self) -> int:
        return self.dividend // self.divisor

    def prompt(self) -> str:
        return f"{self.dividend} ÷ {self.divisor}"

    def pictures(self) -> list[Visual]:
        # The dividend shared into `divisor` equal rows: the answer is the length of a row.
        return [array(self.divisor, self.answer)] if max(self.divisor, self.answer) <= 15 else []

    def hint(self, style: str | None = None) -> str:
        return strategies.divide(self.dividend, self.divisor)
