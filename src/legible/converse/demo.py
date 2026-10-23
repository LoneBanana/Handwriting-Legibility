"""Offline answers for when Claude is not configured (``assistant: "demo"``).

The talk page should still be worth opening without credentials: a learner can
practise listening and reading the avatar's face with a small set of curated,
fact-checked explanations. Each is written in exactly the emotion-tagged format Claude
produces, so it goes through the same segment parser, avatars, and comprehension
checks. A question that matches nothing gets an honest "I'm in demo mode" reply that
suggests topics it does know.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from .segments import Segment, parse_tagged


@dataclass(frozen=True)
class DemoTopic:
    key: str
    label: str                  # what the answer is about (a main-idea choice)
    suggestion: str             # how the fallback reply suggests asking about it
    patterns: tuple[str, ...]   # every pattern must match the question
    answer: str                 # emotion-tagged, in Claude's format
    _compiled: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", tuple(re.compile(p, re.IGNORECASE) for p in self.patterns))

    def matches(self, question: str) -> bool:
        return all(pattern.search(question) for pattern in self._compiled)

    @property
    def segments(self) -> list[Segment]:
        return parse_tagged(self.answer)


TOPICS: tuple[DemoTopic, ...] = (
    DemoTopic(
        key="octopus",
        label="An octopus's three hearts",
        suggestion="how many hearts an octopus has",
        patterns=(r"octop",),
        answer=(
            "[excited] Great question! [surprised] An octopus has three hearts. "
            "[content] Two small hearts pump blood through the gills, where it picks up oxygen. "
            "[content] The third heart pumps that blood around the rest of the body. "
            "[surprised] Octopus blood is blue, because it uses copper to carry oxygen."
        ),
    ),
    DemoTopic(
        key="sky",
        label="Why the sky looks blue",
        suggestion="why the sky is blue",
        patterns=(r"\bsky\b", r"\bblue\b"),
        answer=(
            "[content] Sunlight looks white, but it is really a mix of every colour. "
            "[neutral] When sunlight passes through the air, the air scatters blue light much more than red light. "
            "[surprised] That scattered blue light reaches your eyes from every part of the sky. "
            "[excited] So when you look up on a clear day, the whole sky glows blue!"
        ),
    ),
    DemoTopic(
        key="rainbow",
        label="How rainbows form",
        suggestion="how rainbows form",
        patterns=(r"rainbow",),
        answer=(
            "[excited] A rainbow is sunlight that has been split into its colours. "
            "[neutral] Sunlight is a mix of many colours. "
            "[content] When it shines into raindrops, each colour bends by a slightly different amount. "
            "[surprised] That spreads the light into a band of red, orange, yellow, green, blue and violet. "
            "[content] You see a rainbow when the sun is behind you and the rain is in front of you."
        ),
    ),
    DemoTopic(
        key="dinosaurs",
        label="What happened to the dinosaurs",
        suggestion="what happened to the dinosaurs",
        patterns=(r"dinosaur",),
        answer=(
            "[neutral] Most dinosaurs died out about 66 million years ago. "
            "[scared] A huge rock from space, about 10 kilometres wide, crashed into Earth. "
            "[sad] Dust and smoke blocked the sunlight, so many plants died and many animals starved. "
            "[surprised] But not every dinosaur disappeared. "
            "[excited] Birds are living dinosaurs, so their relatives are still all around us!"
        ),
    ),
    DemoTopic(
        key="moon",
        label="Why the moon seems to change shape",
        suggestion="why the moon changes shape",
        patterns=(r"\bmoon",),
        answer=(
            "[content] The moon does not make its own light. "
            "[neutral] The sun lights up half of the moon, like a lamp shining on half of a ball. "
            "[content] As the moon travels around Earth, we see different amounts of that bright half. "
            "[excited] That is why it seems to grow from a thin crescent to a full moon and back again. "
            "[neutral] One full cycle takes about 29 and a half days."
        ),
    ),
    DemoTopic(
        key="volcano",
        label="How volcanoes erupt",
        suggestion="how volcanoes erupt",
        patterns=(r"volcan|lava|magma|erupt",),
        answer=(
            "[neutral] Deep underground, some rock gets so hot that it melts into magma. "
            "[surprised] Magma is lighter than the solid rock around it, so it slowly pushes upward. "
            "[scared] When enough pressure builds up, it can burst out as lava, ash and gas. "
            "[content] Many volcanoes stay quiet for hundreds of years between eruptions."
        ),
    ),
    DemoTopic(
        key="honey",
        label="How bees make honey",
        suggestion="how bees make honey",
        patterns=(r"\bbees?\b|honey",),
        answer=(
            "[excited] Bees are amazing little chefs! "
            "[content] They collect a sweet liquid called nectar from flowers. "
            "[neutral] Back in the hive, they pass it from bee to bee and fan it with their wings to dry it out. "
            "[surprised] Slowly, the runny nectar turns into thick honey. "
            "[content] Honey can last for a very long time without going bad."
        ),
    ),
    DemoTopic(
        key="sleep",
        label="Why we need sleep",
        suggestion="why we need sleep",
        patterns=(r"\bsleep",),
        answer=(
            "[content] Sleep is when your body and brain recharge. "
            "[neutral] While you sleep, your brain sorts and saves the memories from your day. "
            "[content] Your body also repairs muscles and grows. "
            "[dejected] Without enough sleep, it is harder to focus and to feel calm. "
            "[content] Most children need about 9 to 12 hours of sleep each night."
        ),
    ),
    DemoTopic(
        key="purr",
        label="How and why cats purr",
        suggestion="why cats purr",
        patterns=(r"\bpurr",),
        answer=(
            "[content] A cat purrs by quickly tightening and relaxing muscles in its voice box. "
            "[neutral] This makes the air shake as the cat breathes in and out. "
            "[content] Cats often purr when they feel safe and relaxed. "
            "[surprised] They can also purr when they are hurt or scared, maybe to calm themselves down."
        ),
    ),
    DemoTopic(
        key="penguins",
        label="Why penguins cannot fly",
        suggestion="why penguins can't fly",
        patterns=(r"penguin",),
        answer=(
            "[content] Penguins are birds, but their wings have become stiff flippers. "
            "[neutral] Flippers are too small to lift a heavy penguin into the air. "
            "[excited] Instead, penguins fly underwater! "
            "[surprised] Some penguins can swim faster than most people can run."
        ),
    ),
    DemoTopic(
        key="thunder",
        label="What makes thunder",
        suggestion="what makes thunder",
        patterns=(r"thunder|lightning",),
        answer=(
            "[neutral] Lightning is a giant spark of electricity in the sky. "
            "[surprised] It heats the air around it to about five times hotter than the surface of the sun. "
            "[scared] That super-hot air spreads out so fast that it makes a loud boom. "
            "[content] That boom is thunder. "
            "[neutral] You see the flash first because light travels much faster than sound."
        ),
    ),
    DemoTopic(
        key="ocean",
        label="Why the ocean is salty",
        suggestion="why the sea is salty",
        patterns=(r"ocean|\bseas?\b", r"salt"),
        answer=(
            "[neutral] Rain slowly wears away tiny bits of rock on land. "
            "[content] Rivers carry those bits, including salt, down to the sea. "
            "[neutral] When the sun warms the sea, water rises into the air, but the salt stays behind. "
            "[surprised] This has been happening for millions of years, so the salt has built up. "
            "[content] That is why the ocean tastes salty but rain does not."
        ),
    ),
    DemoTopic(
        key="hiccups",
        label="What causes hiccups",
        suggestion="what causes hiccups",
        patterns=(r"hiccup",),
        answer=(
            "[content] A hiccup starts with your diaphragm, the dome of muscle under your lungs. "
            "[surprised] Sometimes it suddenly jerks and pulls air in very fast. "
            "[neutral] Then your vocal cords snap shut, which stops the air. "
            "[excited] That snap makes the hic sound! "
            "[content] Hiccups usually stop on their own after a few minutes."
        ),
    ),
    DemoTopic(
        key="heart",
        label="How your heart works",
        suggestion="how often your heart beats",
        patterns=(r"\bheart",),
        answer=(
            "[content] Your heart is a strong muscle about the size of your fist. "
            "[neutral] Each beat squeezes blood out to every part of your body. "
            "[surprised] It beats about 100,000 times every day. "
            "[content] When you run or play, it beats faster to send more oxygen to your muscles."
        ),
    ),
    DemoTopic(
        key="leaves",
        label="Why leaves change colour in autumn",
        suggestion="why leaves change colour",
        patterns=(r"lea(f|ves)", r"colou?r|autumn|\bfall\b|orange|yellow|\bred\b"),
        answer=(
            "[content] Leaves are green because of a chemical called chlorophyll. "
            "[neutral] Chlorophyll helps leaves turn sunlight into food for the tree. "
            "[neutral] In autumn the days get shorter, and the tree stops making chlorophyll. "
            "[surprised] As the green fades, yellow and orange colours that were hidden underneath show through. "
            "[excited] Some trees also make new red colours, which is why autumn can look so bright!"
        ),
    ),
    DemoTopic(
        key="planes",
        label="How planes stay in the air",
        suggestion="how planes fly",
        patterns=(r"\b(aero|air)?planes?\b|\bjets?\b", r"\bfl(y|ies|ying)\b|stay up|in the air"),
        answer=(
            "[content] A plane's wings are shaped and tilted so they push air downward as the plane moves. "
            "[surprised] Pushing air down pushes the wings up. "
            "[neutral] That upward push is called lift. "
            "[content] The engines push the plane forward fast enough for the wings to make lots of lift. "
            "[excited] That is how something as heavy as a jumbo jet stays in the air!"
        ),
    ),
    DemoTopic(
        key="snot",
        label="What snot is for",
        suggestion="why we have snot",
        patterns=(r"snot|booger|bogey|bogie|mucus",),
        answer=(
            "[disgusted] Snot might seem gross, but it does an important job. "
            "[content] It is a sticky goo called mucus that lines the inside of your nose. "
            "[neutral] It traps dust, germs and pollen before they can reach your lungs. "
            "[surprised] Your nose makes mucus all day long, even when you are healthy. "
            "[disgusted] Most of it slides down your throat, and you swallow it without noticing!"
        ),
    ),
    DemoTopic(
        key="anger",
        label="What anger is and how to calm it",
        suggestion="what to do when you feel angry",
        patterns=(r"\bangry\b|\banger\b|\bmad\b",),
        answer=(
            "[content] Anger is a normal feeling that everyone has. "
            "[angry] It often shows up when something feels unfair, like when someone pushes in front of you. "
            "[neutral] Your heart beats faster and your muscles get tense. "
            "[content] Taking slow breaths and naming the feeling can help it pass. "
            "[content] Then you can calmly explain what felt unfair."
        ),
    ),
)

_GREETING = re.compile(r"^\W*(hi|hello|hey|hiya|howdy|good (morning|afternoon|evening))\b", re.IGNORECASE)

GREETING_ANSWER = (
    "[excited] Hi there! "
    "[content] I'm your talking guide. "
    "[neutral] I'm in demo mode right now, so I only know about a few topics. "
    "[content] You could ask me why the sky is blue, how volcanoes erupt or what causes hiccups."
)


def match_topic(question: str) -> DemoTopic | None:
    return next((topic for topic in TOPICS if topic.matches(question)), None)


def demo_answer(question: str) -> str:
    """The tagged answer the demo guide gives to ``question``."""
    topic = match_topic(question)
    if topic is not None:
        return topic.answer
    if _GREETING.match(question) and len(question.split()) <= 6:
        return GREETING_ANSWER
    return _fallback_answer(question)


def _fallback_answer(question: str) -> str:
    rng = random.Random(question.strip().lower())  # the same question gets the same suggestions
    first, second, third = (topic.suggestion for topic in rng.sample(TOPICS, 3))
    return (
        "[content] That is a great question. "
        "[dejected] I'm in demo mode right now, so I can only answer a few questions. "
        f"[excited] Try asking me {first}, {second} or {third}!"
    )


async def stream_demo_answer(question: str, *, chunk_size: int = 24) -> AsyncIterator[str]:
    """Yields the demo answer in small chunks, like a streamed reply from Claude."""
    text = demo_answer(question)
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]
        await asyncio.sleep(0)
