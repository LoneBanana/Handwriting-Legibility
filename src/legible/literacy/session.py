"""Builds a flash-card session and checks answers.

A session mixes **read** cards (a word is flashed for ``exposure_ms``, then the learner
picks it from look-alike choices) and **spell** cards (the word is spoken, alone and in
a sentence, and the learner types or writes it). About 70% of the words come from the
current pattern and 30% are spaced review of earlier patterns, chosen by the lowest
predicted recall, the interleaving that keeps old patterns from fading.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

from ..progression import TRACKS
from .curriculum import SKILLS, SKILLS_BY_ID, Skill, Word, skill_index_for, word_bank
from .distractors import distractors
from .memory import MemoryState, recall_probability
from .spelling import analyze, normalize_answer, phonetically_plausible

READING = "literacy.reading"
SPELLING = "literacy.spelling"
REVIEW_SHARE = 0.3

Mode = Literal["read", "spell"]


@dataclass
class CardPlan:
    card_id: str
    word: str
    mode: Mode
    skill: str
    skill_label: str
    graphemes: list[str]
    focus: tuple[int, int] | None
    prompt: str
    choices: list[str] | None
    sentence: str | None
    exposure_ms: int | None
    hint: dict[str, Any] | None

    def public(self) -> dict[str, Any]:
        """What the client sees. (The word is shown on read cards and in feedback anyway;
        spell cards are answered by ear, so the word is part of the card.)"""
        data = asdict(self)
        data["focus"] = list(self.focus) if self.focus else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CardPlan":
        data = dict(data)
        data["focus"] = tuple(data["focus"]) if data.get("focus") else None
        return cls(**data)


@dataclass
class SessionPlan:
    skill: Skill
    cards: list[CardPlan]


def _hint(word: str, hints: int) -> dict[str, Any] | None:
    if hints >= 2:
        return {"length": len(word), "first_letter": word[0]}
    if hints == 1:
        return {"length": len(word), "first_letter": None}
    return None


def _pick(mode: Mode, count: int, skill_index: int, memory: dict[tuple[str, str], MemoryState], now: datetime,
          rng: random.Random, exclude: set[str]) -> list[Word]:
    bank = word_bank()
    current = [w for w in bank[SKILLS[skill_index].id] if w.text not in exclude]
    earlier = [w for skill in SKILLS[:skill_index] for w in bank[skill.id] if w.text not in exclude]

    def risk(word: Word) -> tuple[int, float, float]:
        state = memory.get((word.text, mode))
        seen = state is not None and state.seen > 0
        # Unseen words first (in random order), then the words most likely forgotten.
        return (1 if seen else 0, recall_probability(state, now) if seen else 0.0, rng.random())

    n_review = round(count * REVIEW_SHARE) if earlier else 0
    # Review favours words already practised, ordered by how likely they are forgotten.
    practised = sorted((w for w in earlier if (w.text, mode) in memory),
                       key=lambda w: recall_probability(memory[(w.text, mode)], now))
    fresh_earlier = [w for w in earlier if (w.text, mode) not in memory]
    rng.shuffle(fresh_earlier)
    review = (practised + fresh_earlier)[:n_review]
    new = sorted(current, key=risk)[: count - len(review)]
    if len(new) + len(review) < count:  # a short word list: top up from earlier patterns
        extra = [w for w in practised + fresh_earlier if w not in review][: count - len(new) - len(review)]
        review += extra
    chosen = new + review
    rng.shuffle(chosen)
    return chosen


def build_session(reading_level: int, spelling_level: int, memory: dict[tuple[str, str], MemoryState],
                  now: datetime, rng: random.Random | None = None) -> SessionPlan:
    rng = rng or random.Random()
    skill_index = skill_index_for(reading_level, spelling_level)
    reading = TRACKS[READING].parameter_values(reading_level)
    spelling = TRACKS[SPELLING].parameter_values(spelling_level)

    read_words = _pick("read", int(reading["cards"]), skill_index, memory, now, rng, set())
    spell_words = _pick("spell", int(spelling["cards"]), skill_index, memory, now, rng, {w.text for w in read_words})

    read_cards = [_card("read", word, f"r{i}", reading, spelling) for i, word in enumerate(read_words)]
    spell_cards = [_card("spell", word, f"s{i}", reading, spelling) for i, word in enumerate(spell_words)]
    return SessionPlan(SKILLS[skill_index], _interleave(read_cards, spell_cards))


def _card(mode: Mode, word: Word, card_id: str, reading: dict[str, float], spelling: dict[str, float]) -> CardPlan:
    skill = SKILLS_BY_ID[word.skill]
    common = dict(card_id=card_id, word=word.text, mode=mode, skill=skill.id, skill_label=skill.label,
                  graphemes=list(word.graphemes), focus=word.focus)
    if mode == "read":
        others = distractors(word.text, int(reading["choices"]) - 1, seed=f"{card_id}:{word.text}")
        return CardPlan(**common, prompt="Watch for the word, then choose the one you saw.",
                        choices=[word.text, *others], sentence=None, exposure_ms=int(reading["exposure_ms"]),
                        hint=None)
    return CardPlan(**common, prompt="Listen to the word, then spell it.", choices=None, sentence=word.sentence,
                    exposure_ms=None, hint=_hint(word.text, int(spelling["hints"])))


def _interleave(a: list[CardPlan], b: list[CardPlan]) -> list[CardPlan]:
    """Spreads the two kinds of card evenly through the session (r s r s r s s ...)."""
    total = len(a) + len(b)
    out: list[CardPlan] = []
    i = j = 0
    for k in range(total):
        take_a = j >= len(b) or (i < len(a) and (i + 1) / (len(a) + 1) <= (j + 1) / (len(b) + 1))
        if take_a:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out


# ---------------------------------------------------------------- answers


@dataclass(frozen=True)
class AnswerCheck:
    correct: bool
    expected: str
    phonetically_plausible: bool
    feedback: str
    errors: list[dict[str, Any]]


_PRAISE = {
    "read": ("Yes, that's the word!", "Great reading!", "Spot on, you read it right.", "You've got it!"),
    "spell": ("Perfect spelling!", "Spot on, every letter is right.", "Yes, that's exactly right!"),
}


def check_answer(card: CardPlan, response: str) -> AnswerCheck:
    expected = card.word
    answer = normalize_answer(response)
    if answer == expected:
        options = _PRAISE[card.mode]
        return AnswerCheck(True, expected, False, options[sum(map(ord, card.card_id)) % len(options)], [])

    errors = [e.as_dict() for e in analyze(expected, answer, chosen=card.mode == "read")] if answer else []
    plausible = card.mode == "spell" and phonetically_plausible(expected, answer)
    if not answer:
        feedback = "No problem. Here's the word, so you can see how it goes."
    elif plausible:
        feedback = f"It sounds just right! This word spells that sound with the pattern “{card.skill_label}”."
    elif any(e["type"] == "reversal" for e in errors):
        feedback = "So close! Two letters that look like mirror images got swapped. Take a look below."
    elif card.mode == "read":
        feedback = "Have a careful look at the word below, one sound at a time. Every try helps."
    else:
        feedback = "Good try! Compare your spelling with the word below, one sound at a time."
    return AnswerCheck(False, expected, plausible, feedback, errors)
