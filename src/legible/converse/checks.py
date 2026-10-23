"""Comprehension checks: a few multiple-choice questions about one spoken answer.

Listening questions (main idea, detail, inference) score ``converse.listening``; Claude
writes them when it is available. The emotion question ("How did the avatar feel when
it said ...?") scores ``converse.expressions`` and is built here from the answer's own
emotion tags, so its correct answer is exactly the expression the avatar showed.

When Claude is unavailable, fails, or returns unusable questions, deterministic
listening questions are built from the answer text instead (which sentence was said,
fill the gap). Choices are shuffled with a seed so a check never changes once made,
and answers stay on the server: the client only ever sees prompts and choices.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Protocol, Sequence, TypeVar

from pydantic import BaseModel, Field

from . import demo
from .emotions import FACE_CLUES, FEELING_WORDS, NEUTRAL, emotion_options
from .prompts import CHECK_SYSTEM_PROMPT, check_messages
from .segments import Segment, clean_text, plain_text

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Kinds asked for, in order, as the listening track adds questions (2 at first, 4 at the target).
_KIND_PLAN = ("main_idea", "detail", "inference", "detail")


class StructuredLLM(Protocol):
    """The part of the API's ``LLM`` service that checks need."""

    @property
    def available(self) -> bool: ...

    async def structured(
        self, *, system: str, messages: list[dict], schema: type[T], max_tokens: int = ..., effort: str = ...
    ) -> T: ...


class GeneratedQuestion(BaseModel):
    kind: Literal["main_idea", "detail", "inference"]
    question: str = Field(description="The question, in under 15 simple words.")
    correct_answer: str = Field(description="The one correct choice, in under 10 words.")
    wrong_answers: list[str] = Field(description="Two or three choices that are clearly wrong, each under 10 words.")
    explanation: str = Field(description="One short, friendly sentence pointing to what the guide said.")


class GeneratedCheck(BaseModel):
    questions: list[GeneratedQuestion]


@dataclass
class Question:
    question_id: str
    kind: str  # main_idea | detail | inference | emotion
    prompt: str
    choices: list[str]
    answer: str
    explanation: str

    def public(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "kind": self.kind, "prompt": self.prompt, "choices": list(self.choices)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.public(), "answer": self.answer, "explanation": self.explanation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        return cls(
            question_id=str(data["question_id"]),
            kind=str(data["kind"]),
            prompt=str(data["prompt"]),
            choices=[str(choice) for choice in data["choices"]],
            answer=str(data["answer"]),
            explanation=str(data.get("explanation", "")),
        )


def listening_kinds(count: int) -> list[str]:
    return [_KIND_PLAN[i] if i < len(_KIND_PLAN) else "detail" for i in range(max(count, 0))]


async def build_questions(
    llm: StructuredLLM | None,
    *,
    seed: str,
    learner_question: str,
    segments: Sequence[Segment],
    listening_count: int,
    emotion_choices: int,
    heard: Sequence[str] = (),
) -> list[Question]:
    """``listening_count`` listening questions, plus an emotion question if any segment showed a feeling.

    ``heard`` is what the avatar said earlier in the session, which offline questions
    must not offer as a sentence it did *not* say.
    """
    questions = await listening_questions(
        llm, seed=seed, learner_question=learner_question, segments=segments, count=listening_count, heard=heard
    )
    emotion = emotion_question(seed, segments, emotion_choices)
    if emotion is not None:
        questions.append(emotion)
    return [replace(question, question_id=f"q{i}") for i, question in enumerate(questions, start=1)]


async def listening_questions(
    llm: StructuredLLM | None,
    *,
    seed: str,
    learner_question: str,
    segments: Sequence[Segment],
    count: int,
    heard: Sequence[str] = (),
) -> list[Question]:
    if count <= 0:
        return []
    questions: list[Question] = []
    if llm is not None and llm.available:
        try:
            check = await llm.structured(
                system=CHECK_SYSTEM_PROMPT,
                messages=check_messages(learner_question, plain_text(segments), listening_kinds(count)),
                schema=GeneratedCheck,
                max_tokens=8192,
                effort="medium",
            )
        except Exception as exc:  # whatever went wrong, offline questions are a safe answer
            log.warning("Claude could not write check questions (%s); using offline questions", exc)
        else:
            questions = accept_generated(check.questions, seed)[:count]
    if len(questions) < count:
        questions += offline_questions(seed, learner_question, segments, count - len(questions), heard=heard)
    return questions


def accept_generated(items: Iterable[GeneratedQuestion], seed: str) -> list[Question]:
    """Keeps Claude's questions that have a correct answer and at least two distinct wrong ones."""
    accepted: list[Question] = []
    for item in items:
        prompt = clean_text(item.question)
        answer = clean_text(item.correct_answer)
        wrong = _distinct((clean_text(choice) for choice in item.wrong_answers), exclude=answer)[:3]
        if not prompt or not answer or len(wrong) < 2:
            continue
        explanation = clean_text(item.explanation) or f"The answer was “{answer}”."
        rng = random.Random(f"{seed}:claude:{len(accepted)}")
        accepted.append(_question(item.kind, prompt, answer, wrong, explanation, rng))
    return accepted


# ----------------------------------------------------------------------- offline questions

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d[\d,]*\d|\d")
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
_STOPWORDS = frozenset(
    """a about after all also always an and any are around as at back be because been before but by
    can could did do does down each even every for from get gets go goes had has have he her here him
    his how i if in into is it its just like lot lots make makes many may me more most much must my no
    not now of off on once one only or other our out over really said same say says she should so some
    such than that the their them then there these they thing things this those through to too under up
    us very was way we well were what when where which while who why will with would you your""".split()
)


def offline_questions(
    seed: str, learner_question: str, segments: Sequence[Segment], count: int, *, heard: Sequence[str] = ()
) -> list[Question]:
    """Questions built from the answer text alone, using each sentence at most once.

    Wrong choices come from demo topics the learner has not heard in this session (see
    ``heard``), so they are never something the avatar actually said.
    """
    rng = random.Random(f"{seed}:offline")
    answer_text = plain_text(segments)
    answer_words = set(_content_words(answer_text))
    topic = demo.match_topic(learner_question)
    heard_texts = set(heard)
    others = [other for other in demo.TOPICS if other is not topic]
    unheard = [other for other in others if not any(segment.text in heard_texts for segment in other.segments)]
    others = unheard if len(unheard) >= 3 else others
    pool = [
        segment.text
        for other in others
        for segment in other.segments
        if segment.text not in heard_texts
        and len(_content_words(segment.text)) >= 3
        and len(answer_words & set(_content_words(segment.text))) < 2
    ]

    questions: list[Question] = []
    if topic is not None and plain_text(topic.segments) == answer_text and count > 0:
        questions.append(_main_idea_question(topic, others, rng))
    sentences = [segment.text for segment in segments if len(_content_words(segment.text)) >= 3]
    for i, sentence in enumerate(sentences):
        if len(questions) >= count:
            break
        gap = _gap_question(sentence, answer_words, pool, rng) if i % 2 else None
        question = gap or _said_question(sentence, pool, rng)
        if question is not None:
            questions.append(question)
    return questions[:count]


def _main_idea_question(topic: demo.DemoTopic, others: list[demo.DemoTopic], rng: random.Random) -> Question:
    wrong = [other.label for other in rng.sample(others, 2)]
    about = topic.label[0].lower() + topic.label[1:]
    return _question("main_idea", "What was the answer mostly about?", topic.label, wrong,
                     f"The whole answer was about {about}.", rng)


def _said_question(sentence: str, pool: list[str], rng: random.Random) -> Question | None:
    short = [text for text in pool if len(text) <= max(90, len(sentence))]
    candidates = short if len(short) >= 2 else pool
    if len(candidates) < 2:
        return None
    wrong = rng.sample(candidates, 2)
    return _question("detail", "Which of these sentences was in the answer?", sentence, wrong,
                     "The avatar said exactly this. The other sentences are about different topics.", rng)


def _gap_question(sentence: str, answer_words: set[str], pool: list[str], rng: random.Random) -> Question | None:
    target = _gap_word(sentence)
    if target is None:
        return None
    word = target.group()
    if _is_number(word):
        wrong = _number_distractors(word, rng)
    else:
        wrong = _word_distractors(word, answer_words, pool, rng)
    if len(wrong) < 2:
        return None
    gapped = f"{sentence[: target.start()]}___{sentence[target.end():]}"
    return _question("detail", f"Which word fills the gap? “{gapped}”", word, wrong,
                     f"The avatar said: “{sentence}”", rng)


def _gap_word(sentence: str) -> re.Match[str] | None:
    """A number if the sentence has one, otherwise its longest meaningful word."""
    words = list(_WORD.finditer(sentence))
    numbers = [match for match in words if _is_number(match.group())]
    if numbers:
        return numbers[0]
    content = [match for match in words if len(match.group()) >= 5 and match.group().lower() not in _STOPWORDS]
    return max(content, key=lambda match: len(match.group()), default=None)


def _is_number(word: str) -> bool:
    # "one" is left out: it is usually a pronoun ("the one that ...").
    return word[0].isdigit() or word.lower() in _NUMBER_WORDS[2:]


def _number_distractors(word: str, rng: random.Random) -> list[str]:
    if not word[0].isdigit():
        index = _NUMBER_WORDS.index(word.lower())
        near = [_NUMBER_WORDS[i] for i in (index - 2, index - 1, index + 1, index + 2) if 1 <= i < len(_NUMBER_WORDS)]
        chosen = rng.sample(near, min(2, len(near)))
        return [choice.capitalize() if word[0].isupper() else choice for choice in chosen]
    value = int(word.replace(",", ""))
    if 1000 <= value <= 2100:  # a year: nearby years are the believable mistakes
        options = [value - 10, value + 10, value - 100]
    else:
        options = [value * 2, value * 3, value * 10, value // 2]
    texts = [f"{option:,}" if "," in word or option >= 10_000 else str(option) for option in options if option > 0]
    distinct = [text for text in dict.fromkeys(texts) if text != word]
    return rng.sample(distinct, min(2, len(distinct)))


def _word_distractors(word: str, answer_words: set[str], pool: list[str], rng: random.Random) -> list[str]:
    candidates = sorted(
        {
            candidate
            for text in pool
            for candidate in _content_words(text)
            if abs(len(candidate) - len(word)) <= 2 and candidate not in answer_words and candidate != word.lower()
        }
    )
    chosen = rng.sample(candidates, min(2, len(candidates)))
    return [choice.capitalize() if word[0].isupper() else choice for choice in chosen]


def _content_words(text: str) -> list[str]:
    return [
        word.lower()
        for word in _WORD.findall(text)
        if len(word) >= 3 and not word[0].isdigit() and word.lower() not in _STOPWORDS
    ]


# ----------------------------------------------------------------------- emotion question


def emotion_question(seed: str, segments: Sequence[Segment], choice_count: int) -> Question | None:
    """"How did the avatar feel when it said ...?" about one sentence that showed a feeling."""
    candidates = [segment for segment in segments if segment.emotion != NEUTRAL]
    if not candidates:
        return None
    rng = random.Random(f"{seed}:emotion")
    segment = rng.choice(candidates)
    options = emotion_options(segment.emotion, max(choice_count, 2), rng)
    answer = FEELING_WORDS[segment.emotion]
    return Question(
        question_id="",
        kind="emotion",
        prompt=f"How did the avatar feel when it said “{_quote(segment.text)}”?",
        choices=[FEELING_WORDS[option] for option in options],
        answer=answer,
        explanation=f"The avatar felt {answer}. You can tell from {FACE_CLUES[segment.emotion]}.",
    )


def _quote(text: str, max_words: int = 14) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(",;:") + "…"
    return text.rstrip(".")


# ------------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class CheckScore:
    results: list[dict[str, Any]]  # question_id, correct, expected, explanation
    correct: int
    total: int
    listening: float | None        # share of listening questions right (None: there were none)
    expressions: float | None      # share of emotion questions right (None: there were none)

    @property
    def score(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def perfect(self) -> bool:
        return self.total > 0 and self.correct == self.total


def score_check(questions: Sequence[Question], answers: dict[str, str]) -> CheckScore:
    """Scores the learner's choices; an unanswered question counts as wrong."""
    results: list[dict[str, Any]] = []
    listening: list[bool] = []
    expressions: list[bool] = []
    for question in questions:
        given = answers.get(question.question_id)
        correct = given is not None and _key(given) == _key(question.answer)
        results.append(
            {"question_id": question.question_id, "correct": correct, "expected": question.answer,
             "explanation": question.explanation}
        )
        (expressions if question.kind == "emotion" else listening).append(correct)
    return CheckScore(
        results=results,
        correct=sum(result["correct"] for result in results),
        total=len(results),
        listening=sum(listening) / len(listening) if listening else None,
        expressions=sum(expressions) / len(expressions) if expressions else None,
    )


# ------------------------------------------------------------------------------- helpers


def _question(kind: str, prompt: str, answer: str, wrong: list[str], explanation: str, rng: random.Random) -> Question:
    choices = [answer, *wrong]
    rng.shuffle(choices)
    return Question(question_id="", kind=kind, prompt=prompt, choices=choices, answer=answer, explanation=explanation)


def _distinct(values: Iterable[str], *, exclude: str) -> list[str]:
    seen = {_key(exclude)}
    out: list[str] = []
    for value in values:
        if value and _key(value) not in seen:
            seen.add(_key(value))
            out.append(value)
    return out


def _key(text: str) -> str:
    return " ".join(text.split()).casefold().rstrip(".!?")
