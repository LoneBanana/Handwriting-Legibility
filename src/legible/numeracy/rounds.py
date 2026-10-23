"""Plans a practice round: mostly the current skill, plus interleaved review.

* About 70% of the problems practise the learner's current skill and about 30% review
  earlier skills. Mixing old and new (interleaving) and coming back to facts after a
  gap (spacing) both make facts last longer than practising one kind at a time.
* Within each part, facts are taken in order of :func:`.memory.priority`: often-missed
  or slow facts first, then new and long-unseen ones, well-known ones last.
* No fact appears twice in a round (turn-around facts count as the same fact).
* A picture appears with probability ``visual_support`` / 100 when the skill has one
  for the fact (concrete → pictorial → abstract). Counting dots and reading a number
  line always show their picture.
* Every problem has the same time limit: the track's ``time_limit_ms`` × the current
  skill's ``time_factor``.
* The round is shuffled, so review is spread through it.

Answers, tolerances and hints stay in the server-side payload (:meth:`RoundPlan.to_payload`);
only :meth:`Problem.public` is sent to the client.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..progression import TRACKS
from .facts import AnswerFormat, Fact, Kind
from .ladder import LADDER, Skill, skill_index
from .memory import FactStats, priority
from .visuals import Visual

TRACK_KEY = "numeracy.fluency"
REVIEW_SHARE = 0.3


@dataclass(frozen=True)
class Problem:
    problem_id: str
    skill: str             # the skill whose pictures and hints this problem uses
    fact: Fact             # as shown (turn-around facts may have swapped their numbers)
    visual: Visual | None
    hint: str
    review: bool           # interleaved from an earlier skill

    @property
    def kind(self) -> Kind:
        return self.fact.kind

    @property
    def prompt(self) -> str:
        return self.fact.prompt()

    @property
    def answer_format(self) -> AnswerFormat:
        return self.fact.answer_format

    @property
    def choices(self) -> list[str] | None:
        return self.fact.choices()

    def public(self) -> dict[str, Any]:
        """What the client sees: the ``Problem`` type of the HTTP contract, without the answer."""
        return {
            "problem_id": self.problem_id,
            "kind": self.kind,
            "prompt": self.prompt,
            "visual": dict(self.visual) if self.visual is not None else None,
            "answer_format": self.answer_format,
            "choices": self.choices,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.public(),
            "skill": self.skill,
            "fact": self.fact.key,
            "answer": self.fact.answer,
            "tolerance": self.fact.tolerance,
            "hint": self.hint,
            "review": self.review,
        }


@dataclass(frozen=True)
class RoundSettings:
    skill: Skill
    skill_index: int
    problems: int
    time_limit_ms: int
    visual_chance: float  # 0..1


@dataclass(frozen=True)
class RoundPlan:
    level: int
    skill: Skill
    time_limit_ms: int
    problems: tuple[Problem, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "track": TRACK_KEY,
            "level": self.level,
            "skill": self.skill.id,
            "time_limit_ms": self.time_limit_ms,
            "problems": [problem.to_payload() for problem in self.problems],
        }


def round_settings(level: int) -> RoundSettings:
    values = TRACKS[TRACK_KEY].parameter_values(level)
    index = skill_index(level)
    skill = LADDER[index]
    return RoundSettings(
        skill=skill,
        skill_index=index,
        problems=int(values["problems"]),
        time_limit_ms=round(values["time_limit_ms"] * skill.time_factor),
        visual_chance=min(max(values["visual_support"] / 100, 0.0), 1.0),
    )


def review_count(problems: int, has_earlier_skills: bool) -> int:
    return math.floor(problems * REVIEW_SHARE + 0.5) if has_earlier_skills else 0


def plan_round(level: int, memory: Mapping[str, FactStats], *, rng: random.Random, now: datetime) -> RoundPlan:
    """The round for a ``numeracy.fluency`` level, given the learner's history of each fact (by key)."""
    settings = round_settings(level)
    skill = settings.skill
    earlier = LADDER[: settings.skill_index]
    reviews = review_count(settings.problems, bool(earlier))
    taken: set[str] = set()

    def most_needed(candidates: Iterable[tuple[Skill, Fact]], count: int) -> list[tuple[Skill, Fact]]:
        return _most_needed(candidates, count, memory, taken, now=now, time_limit_ms=settings.time_limit_ms, rng=rng)

    current = most_needed(((skill, fact) for fact in skill.facts), settings.problems - reviews)
    # Review also fills any places the current skill is too small for.
    review = most_needed(_review_candidates(earlier), settings.problems - len(current))
    picks = [(s, f, False) for s, f in current] + [(s, f, True) for s, f in review]
    while len(picks) < settings.problems:  # never happens with the shipped ladder (see tests)
        picks.append((skill, rng.choice(skill.facts), False))

    ids: set[str] = set()
    problems = [
        build_problem(
            from_skill,
            fact,
            rng=rng,
            show_picture=rng.random() < settings.visual_chance,
            review=is_review,
            problem_id=_new_id(rng, ids),
        )
        for from_skill, fact, is_review in picks
    ]
    rng.shuffle(problems)
    return RoundPlan(level=level, skill=skill, time_limit_ms=settings.time_limit_ms, problems=tuple(problems))


def build_problem(
    skill: Skill,
    fact: Fact,
    *,
    rng: random.Random,
    show_picture: bool,
    review: bool = False,
    problem_id: str,
) -> Problem:
    """Turns a fact into a problem with the skill's kind of picture (if shown) and its hint style."""
    shown = fact.presented(rng)
    pictures = [picture for picture in shown.pictures() if picture["type"] in skill.visuals]
    visual = rng.choice(pictures) if pictures and (shown.needs_picture or show_picture) else None
    if shown.needs_picture and visual is None:
        raise ValueError(f"skill {skill.id!r} has no picture for {shown.key!r}")
    return Problem(
        problem_id=problem_id,
        skill=skill.id,
        fact=shown,
        visual=visual,
        hint=shown.hint(skill.hint_style),
        review=review,
    )


def _review_candidates(earlier: Sequence[Skill]) -> list[tuple[Skill, Fact]]:
    """Every fact of the earlier skills, once each, from the skill that most recently taught it."""
    seen: set[str] = set()
    candidates = []
    for skill in reversed(earlier):
        for fact in skill.facts:
            if fact.key not in seen:
                seen.add(fact.key)
                candidates.append((skill, fact))
    return candidates


def _most_needed(
    candidates: Iterable[tuple[Skill, Fact]],
    count: int,
    memory: Mapping[str, FactStats],
    taken: set[str],
    *,
    now: datetime,
    time_limit_ms: float,
    rng: random.Random,
) -> list[tuple[Skill, Fact]]:
    """The ``count`` candidates with the highest practice priority, skipping facts already in the round."""
    if count <= 0:
        return []
    ranked = [
        (priority(memory.get(fact.key), now=now, time_limit_ms=time_limit_ms, rng=rng), skill, fact)
        for skill, fact in candidates
        if fact.key not in taken
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for _, skill, fact in ranked:
        if len(chosen) == count:
            break
        if fact.key not in taken:
            taken.add(fact.key)
            chosen.append((skill, fact))
    return chosen


def _new_id(rng: random.Random, used: set[str]) -> str:
    """A short id, unique within the round (the web app also seeds its dot layouts with it)."""
    while True:
        problem_id = f"{rng.getrandbits(48):012x}"
        if problem_id not in used:
            used.add(problem_id)
            return problem_id
