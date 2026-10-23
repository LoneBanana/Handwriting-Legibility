"""Grading a round: accuracy first, then speed.

Each problem earns credit:

* 1.0 when correct and answered within the round's time limit ("fluent"),
* 0.7 when correct but slower,
* 0 when wrong or not answered.

The round's score is the mean credit, so it reaches 1.0 (a perfect round, which
promotes the track) only when every problem is quick and correct. A round that is all
correct but slow scores 0.7, above the track's 0.6 "struggling" line: being accurate is
never treated as struggling, and the level simply waits until speed comes.

Typed answers are read leniently (surrounding spaces, a leading plus or any kind of
minus sign, non-ASCII digits): a learner should not lose a fact to formatting.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FLUENT_CREDIT = 1.0
SLOW_CREDIT = 0.7

# Minus-like characters keyboards and phones produce (true minus, dashes, full-width forms).
_SIGNS = str.maketrans({c: "-" for c in "−‐‑‒–—﹣－"} | {"＋": "+"})
_INTEGER = re.compile(r"([+-]?)\s*(\d+)")


def parse_integer(text: str) -> int | None:
    """``" 12 "`` → 12, ``"−3"`` → -3, ``"７"`` → 7; anything else → None."""
    match = _INTEGER.fullmatch(text.translate(_SIGNS).strip())
    if match is None:
        return None
    value = int(match.group(2))
    return -value if match.group(1) == "-" else value


@dataclass(frozen=True)
class AnswerKey:
    """The server-side half of a problem, read back from the round's payload."""

    problem_id: str
    skill: str
    fact: str
    answer_format: str
    answer: int | str
    tolerance: int
    choices: tuple[str, ...] | None
    hint: str

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> AnswerKey:
        choices = data.get("choices")
        return cls(
            problem_id=str(data["problem_id"]),
            skill=str(data["skill"]),
            fact=str(data["fact"]),
            answer_format=str(data["answer_format"]),
            answer=data["answer"],
            tolerance=int(data.get("tolerance") or 0),
            choices=tuple(choices) if choices is not None else None,
            hint=str(data.get("hint") or ""),
        )

    @property
    def expected(self) -> str:
        """The exact answer (for an estimate, the true value), as shown in the results."""
        return str(self.answer)

    def accepts(self, response: str) -> bool:
        if self.answer_format == "choice":
            return _matching_choice(response, self.choices or ()) == str(self.answer)
        value = parse_integer(response)
        return value is not None and abs(value - int(self.answer)) <= self.tolerance


def _matching_choice(response: str, choices: Sequence[str]) -> str | None:
    """The choice the response names (by text, or by number for numeric choices), if any."""
    text = response.strip()
    if text in choices:
        return text
    number = parse_integer(text)
    if number is None:
        return None
    return next((choice for choice in choices if parse_integer(choice) == number), None)


@dataclass(frozen=True)
class Graded:
    problem_id: str
    fact: str
    skill: str
    answered: bool
    correct: bool
    fluent: bool                 # correct and within the time limit
    latency_ms: float | None
    expected: str
    strategy_hint: str | None    # only for wrong or slow answers

    @property
    def credit(self) -> float:
        if self.fluent:
            return FLUENT_CREDIT
        return SLOW_CREDIT if self.correct else 0.0


def grade(key: AnswerKey, response: str | None, latency_ms: float | None, time_limit_ms: float) -> Graded:
    """Grades one answer; ``response`` is None when the problem was not answered."""
    correct = response is not None and key.accepts(response)
    fluent = correct and latency_ms is not None and latency_ms <= time_limit_ms
    return Graded(
        problem_id=key.problem_id,
        fact=key.fact,
        skill=key.skill,
        answered=response is not None,
        correct=correct,
        fluent=fluent,
        latency_ms=latency_ms,
        expected=key.expected,
        strategy_hint=None if fluent else key.hint or None,
    )


@dataclass(frozen=True)
class RoundScore:
    score: float     # mean credit, 0..1
    perfect: bool    # every problem quick and correct
    correct: int
    fluent: int
    total: int


def score_round(graded: Sequence[Graded]) -> RoundScore:
    total = len(graded)
    if total == 0:
        return RoundScore(score=0.0, perfect=False, correct=0, fluent=0, total=0)
    fluent = sum(g.fluent for g in graded)
    return RoundScore(
        score=sum(g.credit for g in graded) / total,
        perfect=fluent == total,
        correct=sum(g.correct for g in graded),
        fluent=fluent,
        total=total,
    )
