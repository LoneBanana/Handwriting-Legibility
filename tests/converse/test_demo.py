import re

import pytest

from legible.converse.demo import GREETING_ANSWER, TOPICS, demo_answer, match_topic, stream_demo_answer
from legible.converse.emotions import EMOTIONS
from legible.converse.segments import parse_tagged

_TAG = re.compile(r"\[(\w+)\]")


@pytest.mark.parametrize("topic", TOPICS, ids=lambda topic: topic.key)
def test_every_demo_answer_is_well_formed(topic):
    tags = _TAG.findall(topic.answer)
    segments = topic.segments
    assert set(tags) <= set(EMOTIONS)                   # only vocabulary tags
    assert len(segments) == len(tags)                   # every sentence has its own tag
    assert 3 <= len(segments) <= 7
    assert all(len(segment.text.split()) <= 22 for segment in segments)  # short, spoken sentences
    assert len({segment.emotion for segment in segments}) >= 2           # something to read on the face
    assert topic.matches(topic.suggestion)              # the fallback's suggestion leads here


@pytest.mark.parametrize(
    ("question", "key"),
    [
        ("How many hearts does an octopus have?", "octopus"),
        ("why is the sky blue", "sky"),
        ("What makes a RAINBOW?", "rainbow"),
        ("How do airplanes stay up?", "planes"),
        ("Why is the sea salty?", "ocean"),
        ("why did the chicken cross the road", None),
        ("Is the sky big?", None),
    ],
)
def test_match_topic(question, key):
    topic = match_topic(question)
    assert (topic.key if topic else None) == key


def test_unknown_question_says_demo_mode_and_suggests_real_topics():
    answer = demo_answer("What is the capital of France?")
    assert "demo mode" in answer
    assert answer == demo_answer("what is the capital of france?  ")  # stable for the same question
    suggested = [topic for topic in TOPICS if topic.suggestion in answer]
    assert len(suggested) == 3
    assert all(segment.emotion in EMOTIONS for segment in parse_tagged(answer))


def test_greeting_introduces_the_guide():
    assert demo_answer("Hello!") == GREETING_ANSWER
    assert "demo mode" in GREETING_ANSWER
    assert demo_answer("hi, why is the sky blue?") == match_topic("sky blue").answer


@pytest.mark.anyio
async def test_stream_yields_the_answer_in_small_chunks():
    chunks = [chunk async for chunk in stream_demo_answer("Tell me about volcanoes", chunk_size=10)]
    assert len(chunks) > 5 and all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) == demo_answer("Tell me about volcanoes")
