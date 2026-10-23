import random

import pytest

from legible.api.services.llm import LLMError
from legible.converse import demo
from legible.converse.checks import (
    GeneratedCheck,
    Question,
    accept_generated,
    build_questions,
    emotion_question,
    listening_kinds,
    offline_questions,
    score_check,
)
from legible.converse.emotions import FAMILIES, FEELING_WORDS, emotion_options
from legible.converse.prompts import CHECK_SYSTEM_PROMPT
from legible.converse.segments import parse_tagged

from .fakes import FakeLLM

ANSWER = parse_tagged(
    "[excited] Great question! [surprised] An octopus has three hearts. "
    "[content] Two small hearts pump blood through the gills. "
    "[neutral] The third heart pumps blood to the rest of the body."
)

GOOD_CHECK = {
    "questions": [
        {
            "kind": "main_idea",
            "question": "What was the answer mostly about?",
            "correct_answer": "An octopus's hearts",
            "wrong_answers": ["Shark teeth", "Whale songs"],
            "explanation": "The guide talked about the octopus's three hearts.",
        },
        {
            "kind": "detail",
            "question": "How many hearts does an octopus have?",
            "correct_answer": "Three",
            "wrong_answers": ["One", "Two", "Five"],
            "explanation": "The guide said an octopus has three hearts.",
        },
    ]
}


class UnavailableLLM:
    available = False


def by_kind(questions: list[Question]) -> dict[str, list[Question]]:
    out: dict[str, list[Question]] = {}
    for question in questions:
        out.setdefault(question.kind, []).append(question)
    return out


@pytest.mark.anyio
async def test_claude_questions_plus_an_emotion_question():
    llm = FakeLLM(check=GOOD_CHECK)
    questions = await build_questions(
        llm, seed="turn-1", learner_question="How many hearts?", segments=ANSWER, listening_count=2, emotion_choices=4
    )
    assert [question.question_id for question in questions] == ["q1", "q2", "q3"]
    assert [question.kind for question in questions] == ["main_idea", "detail", "emotion"]
    for question in questions:
        assert question.answer in question.choices
        assert len(question.choices) == len(set(question.choices))
    assert sorted(questions[1].choices) == ["Five", "One", "Three", "Two"]

    call = llm.structured_calls[0]
    assert call["system"] == CHECK_SYSTEM_PROMPT
    assert "main_idea, detail" in call["messages"][0]["content"]
    assert "An octopus has three hearts." in call["messages"][0]["content"]


@pytest.mark.anyio
async def test_choices_are_shuffled_the_same_way_every_time():
    def make():
        return build_questions(
            FakeLLM(check=GOOD_CHECK), seed="turn-1", learner_question="q", segments=ANSWER,
            listening_count=2, emotion_choices=6,
        )

    first, second = await make(), await make()
    assert [q.choices for q in first] == [q.choices for q in second]
    other = await build_questions(
        FakeLLM(check=GOOD_CHECK), seed="turn-2", learner_question="q", segments=ANSWER,
        listening_count=2, emotion_choices=6,
    )
    assert [q.choices for q in other] != [q.choices for q in first]  # a different turn, a different order


def test_unusable_claude_questions_are_dropped():
    check = {
        "questions": [
            {**GOOD_CHECK["questions"][1], "wrong_answers": ["Three", "three."]},   # no real wrong answer
            {**GOOD_CHECK["questions"][1], "correct_answer": "  "},                # no answer
            {**GOOD_CHECK["questions"][1], "wrong_answers": ["One", "One", "Two"]},  # duplicates collapse
        ]
    }
    accepted = accept_generated(GeneratedCheck.model_validate(check).questions, "seed")
    assert len(accepted) == 1 and sorted(accepted[0].choices) == ["One", "Three", "Two"]


@pytest.mark.anyio
@pytest.mark.parametrize("llm", [UnavailableLLM(), FakeLLM(check=LLMError("down")), FakeLLM(check=ValueError("bad"))])
async def test_offline_questions_when_claude_is_missing_or_fails(llm):
    topic = demo.match_topic("octopus")
    questions = await build_questions(
        llm, seed="turn-9", learner_question="How many hearts does an octopus have?",
        segments=topic.segments, listening_count=3, emotion_choices=3,
    )
    kinds = by_kind(questions)
    assert len(questions) == 4 and len(kinds["emotion"]) == 1
    assert kinds["main_idea"][0].answer == topic.label  # the demo answer's topic is known
    for question in questions:
        assert question.answer in question.choices and len(set(question.choices)) == len(question.choices) >= 3


@pytest.mark.anyio
async def test_too_few_claude_questions_are_topped_up_offline():
    check = {"questions": GOOD_CHECK["questions"][:1]}
    questions = await build_questions(
        FakeLLM(check=check), seed="t", learner_question="octopus hearts", segments=ANSWER,
        listening_count=3, emotion_choices=3,
    )
    listening = [question for question in questions if question.kind != "emotion"]
    assert len(listening) == 3 and listening[0].answer == "An octopus's hearts"


def test_offline_questions_use_the_answer_and_never_what_was_heard():
    segments = parse_tagged(
        "[content] Paris is the capital city of France. [surprised] The Eiffel Tower was finished in 1889. "
        "[neutral] About two million people live inside Paris itself."
    )
    heard = [segment.text for topic in demo.TOPICS[:12] for segment in topic.segments]
    questions = offline_questions("seed", "What is the capital of France?", segments, 3, heard=heard)
    assert len(questions) == 3
    said = [segment.text for segment in segments]
    for question in questions:
        assert question.kind == "detail"
        wrong = [choice for choice in question.choices if choice != question.answer]
        assert not set(wrong) & set(heard) and not set(wrong) & set(said)
    gap = next(question for question in questions if question.prompt.startswith("Which word fills the gap?"))
    assert gap.answer == "1889" and "___" in gap.prompt


def test_offline_questions_need_sentences_with_substance():
    assert offline_questions("seed", "hi", parse_tagged("[excited] Hi there! [content] Good question."), 2) == []


def test_emotion_question_uses_feeling_words_and_distinct_families():
    question = emotion_question("seed", ANSWER, 6)
    assert question is not None and question.kind == "emotion"
    assert question.prompt.startswith("How did the avatar feel when it said “")
    assert len(question.choices) == 6 and question.answer in question.choices
    assert set(question.choices) <= set(FEELING_WORDS.values())
    emotions = [next(e for e, word in FEELING_WORDS.items() if word == choice) for choice in question.choices]
    assert len({FAMILIES[emotion] for emotion in emotions}) == 6
    assert question.explanation.startswith(f"The avatar felt {question.answer}.")


def test_no_emotion_question_when_every_sentence_is_neutral():
    assert emotion_question("seed", parse_tagged("[neutral] Plain fact one. Plain fact two."), 3) is None


def test_emotion_options_include_the_target_once():
    for target in ("content", "dejected", "contempt", "neutral"):
        for count in (3, 4, 5, 6):
            options = emotion_options(target, count, random.Random(f"{target}{count}"))
            assert len(options) == count and options.count(target) == 1
            assert len({FAMILIES[option] for option in options}) == count


def test_listening_kinds_grow_with_the_track():
    assert listening_kinds(2) == ["main_idea", "detail"]
    assert listening_kinds(3) == ["main_idea", "detail", "inference"]
    assert listening_kinds(4) == ["main_idea", "detail", "inference", "detail"]


def test_score_check_splits_listening_and_expressions():
    questions = [
        Question("q1", "main_idea", "?", ["a", "b", "c"], "a", "because a"),
        Question("q2", "detail", "?", ["x", "y", "z"], "Three hearts.", "because"),
        Question("q3", "emotion", "?", ["excited", "sad", "angry"], "excited", "smile"),
    ]
    scored = score_check(questions, {"q1": "a", "q2": " three   hearts ", "q3": "sad"})
    assert [result["correct"] for result in scored.results] == [True, True, False]
    assert scored.listening == 1.0 and scored.expressions == 0.0
    assert scored.score == pytest.approx(2 / 3) and not scored.perfect
    assert scored.results[2] == {"question_id": "q3", "correct": False, "expected": "excited", "explanation": "smile"}

    unanswered = score_check(questions[:2], {})
    assert unanswered.listening == 0.0 and unanswered.expressions is None
    assert score_check([], {}).perfect is False


def test_questions_round_trip_through_storage_without_leaking_answers():
    question = Question("q1", "detail", "Which?", ["a", "b"], "a", "why")
    assert Question.from_dict(question.to_dict()) == question
    assert "answer" not in question.public() and "explanation" not in question.public()
