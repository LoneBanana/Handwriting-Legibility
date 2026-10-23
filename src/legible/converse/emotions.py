"""The emotion vocabulary shared by Claude, the avatars, and the facial-expression panel.

It is exactly the set of values Tavus Phoenix-4 faces accept in ``<emotion value="..."/>``
echo tags, so a tag Claude writes can drive a realistic face unchanged. Learners never
see the raw names: :data:`FEELING_WORDS` are the everyday words used in comprehension
questions, and :data:`FACE_CLUES` say what each expression looks like (the EMFACS
prototypes in docs/ARCHITECTURE.md) so a wrong answer comes with something to look for.
"""

from __future__ import annotations

import random

NEUTRAL = "neutral"

EMOTIONS: tuple[str, ...] = (
    "neutral",
    "content",
    "excited",
    "elated",
    "surprised",
    "sad",
    "dejected",
    "scared",
    "angry",
    "disgusted",
    "contempt",
)

FEELING_WORDS: dict[str, str] = {
    "neutral": "neutral (no strong feeling)",
    "content": "calm and happy",
    "excited": "excited",
    "elated": "overjoyed",
    "surprised": "surprised",
    "sad": "sad",
    "dejected": "discouraged",
    "scared": "scared",
    "angry": "angry",
    "disgusted": "disgusted",
    "contempt": "scornful",
}

FACE_CLUES: dict[str, str] = {
    "neutral": "a relaxed face, with no strong movement of the eyebrows or mouth",
    "content": "a gentle smile, with the cheeks lifted a little",
    "excited": "a big smile, raised cheeks and wide, bright eyes",
    "elated": "a huge smile, with the cheeks pushed up so the eyes crinkle",
    "surprised": "eyebrows raised high, eyes wide open and the jaw dropped",
    "sad": "the inner eyebrows raised and the corners of the mouth pulled down",
    "dejected": "the inner eyebrows raised and the mouth corners pulled down, with a tired, low-energy look",
    "scared": "eyebrows raised and pulled together, eyes wide and lips stretched sideways",
    "angry": "eyebrows pulled down and together, a hard stare and lips pressed tight",
    "disgusted": "a wrinkled nose and a raised upper lip",
    "contempt": "one corner of the mouth pulled up in a one-sided smirk",
}

# Emotions that share a facial prototype differ mostly in strength (excited vs.
# overjoyed). As expressions get subtler that difference becomes unreadable, so a
# question never offers two feelings from the same family.
FAMILIES: dict[str, str] = {
    "neutral": "neutral",
    "content": "happy",
    "excited": "happy",
    "elated": "happy",
    "surprised": "surprised",
    "sad": "sad",
    "dejected": "sad",
    "scared": "scared",
    "angry": "angry",
    "disgusted": "disgusted",
    "contempt": "contempt",
}

_KNOWN = frozenset(EMOTIONS)


def normalize_emotion(name: str | None) -> str:
    """The vocabulary entry for a tag Claude wrote; anything unrecognised is neutral."""
    if not name:
        return NEUTRAL
    key = name.strip().lower()
    return key if key in _KNOWN else NEUTRAL


def feeling_word(emotion: str) -> str:
    return FEELING_WORDS[normalize_emotion(emotion)]


def emotion_options(target: str, count: int, rng: random.Random) -> list[str]:
    """``count`` emotions (target included, in random order), each from a different family."""
    target = normalize_emotion(target)
    by_family: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        family = FAMILIES[emotion]
        if family != FAMILIES[target]:
            by_family.setdefault(family, []).append(emotion)
    families = sorted(by_family)
    rng.shuffle(families)
    distractors = [rng.choice(by_family[family]) for family in families[: max(count - 1, 0)]]
    options = [target, *distractors]
    rng.shuffle(options)
    return options
