"""Phase 3: math-fact fluency for dyscalculia.

* :mod:`.ladder` — 26 skills from seeing small amounts to fluent times tables;
  the ``numeracy.fluency`` level picks the skill (12 levels per skill).
* :mod:`.facts` and :mod:`.strategies` — each fact's prompt, answer, pictures, and the
  strategy hint for a wrong or slow answer.
* :mod:`.rounds` — a round is ~70% current skill and ~30% interleaved review,
  weakest facts first, with pictures that fade as the track's visual support falls.
* :mod:`.scoring` and :mod:`.memory` — accuracy first, then speed: only a round of
  quick, correct answers is perfect, and every answer updates the fact's history.

Everything here is pure (no I/O); ``legible.api.routers.numeracy`` stores rounds and
fact memory.
"""

from .facts import Fact
from .ladder import LADDER, LEVELS_PER_SKILL, SKILLS, Skill, skill_for_level, skill_index
from .memory import FactStats, priority, record_attempt
from .rounds import TRACK_KEY, Problem, RoundPlan, build_problem, plan_round, round_settings
from .scoring import AnswerKey, Graded, RoundScore, grade, parse_integer, score_round

__all__ = [
    "LADDER",
    "LEVELS_PER_SKILL",
    "SKILLS",
    "TRACK_KEY",
    "AnswerKey",
    "Fact",
    "FactStats",
    "Graded",
    "Problem",
    "RoundPlan",
    "RoundScore",
    "Skill",
    "build_problem",
    "grade",
    "parse_integer",
    "plan_round",
    "priority",
    "record_attempt",
    "round_settings",
    "score_round",
    "skill_for_level",
    "skill_index",
]
