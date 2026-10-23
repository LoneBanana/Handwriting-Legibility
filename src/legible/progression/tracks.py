"""Every progression track in the app and the difficulty parameters it controls.

Parameters are pure functions of the level (see :mod:`.curves`), so the level is the
only state stored per learner. Modules read their settings from here, e.g.
``TRACKS["converse.listening"].parameter_values(level)["words_per_minute"]``.
"""

from __future__ import annotations

from .curves import Curve
from .engine import TrackSpec

HANDWRITING_LEGIBILITY = TrackSpec(
    key="handwriting.legibility",
    module="handwriting",
    label="Handwriting clarity",
    description="How clearly Legible can read your own handwriting. The goal creeps up as your writing gets clearer.",
    parameters={
        # A transcription counts as perfect when its legibility reaches the goal.
        "goal": Curve(60, 90, "%", "Clarity goal", shape="linear"),
    },
    levels_to_target=150,
    weekly_goal=3,
    promotion_note="Your clarity goal rises a little.",
)

LITERACY_READING = TrackSpec(
    key="literacy.reading",
    module="literacy",
    label="Reading",
    description="Recognising words accurately from a quick flash, using the sound patterns you have learned.",
    parameters={
        "exposure_ms": Curve(2000, 500, "ms", "Word flash time"),
        "choices": Curve(3, 5, "choices", "Answer choices", shape="linear", integer=True),
        "cards": Curve(5, 8, "cards", "Reading cards per session", shape="linear", integer=True),
    },
    promotion_note="Words will flash a little faster next time.",
)

LITERACY_SPELLING = TrackSpec(
    key="literacy.spelling",
    module="literacy",
    label="Spelling",
    description="Spelling words you hear, with hints that fade as the patterns stick.",
    parameters={
        "cards": Curve(4, 8, "cards", "Spelling cards per session", shape="linear", integer=True),
        # 2 = letter boxes and the first letter, 1 = letter boxes only, 0 = no hints.
        "hints": Curve(2, 0, "", "Spelling hints", shape="linear", integer=True),
    },
    promotion_note="You are ready for slightly less help.",
)

NUMERACY_FLUENCY = TrackSpec(
    key="numeracy.fluency",
    module="numeracy",
    label="Math fact fluency",
    description="Answering number facts correctly first, then quickly, from counting dots up to times tables.",
    parameters={
        # Each skill scales this (x0.6 for seeing dots up to x1.5 for times tables and division).
        "time_limit_ms": Curve(10000, 2500, "ms", "Base time per problem"),
        "problems": Curve(8, 15, "problems", "Problems per round", shape="linear", integer=True),
        "visual_support": Curve(100, 0, "%", "Picture support", shape="linear"),
    },
    promotion_note="The timer will be a touch shorter next round.",
)

CONVERSE_LISTENING = TrackSpec(
    key="converse.listening",
    module="converse",
    label="Listening",
    description="Following spoken explanations, from slow and captioned to natural speed without captions.",
    parameters={
        "words_per_minute": Curve(110, 180, "wpm", "Speaking speed"),
        # >= 67: full captions, >= 34: captions appear after each sentence is spoken, below: off.
        "caption_support": Curve(100, 0, "%", "Caption support", shape="linear"),
        "questions": Curve(2, 4, "questions", "Questions per check", shape="linear", integer=True),
    },
    promotion_note="The avatar will speak a touch faster next time.",
)

CONVERSE_EXPRESSIONS = TrackSpec(
    key="converse.expressions",
    module="converse",
    label="Reading faces",
    description="Recognising feelings from facial expressions, from big and clear to small and subtle.",
    parameters={
        "expression_intensity": Curve(100, 30, "%", "Expression strength"),
        "emotion_choices": Curve(3, 6, "choices", "Feelings to choose from", shape="linear", integer=True),
    },
    promotion_note="Expressions will be a little more subtle next time.",
)

TRACKS: dict[str, TrackSpec] = {
    spec.key: spec
    for spec in (
        HANDWRITING_LEGIBILITY,
        LITERACY_READING,
        LITERACY_SPELLING,
        NUMERACY_FLUENCY,
        CONVERSE_LISTENING,
        CONVERSE_EXPRESSIONS,
    )
}


def caption_mode(caption_support: float) -> str:
    """Maps the listening track's caption support (0..100) to the UI caption mode."""
    if caption_support >= 67:
        return "full"
    if caption_support >= 34:
        return "delayed"
    return "off"
