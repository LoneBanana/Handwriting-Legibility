"""Phase 4: conversation practice for auditory processing, language processing, and
nonverbal learning disorders.

The learner asks a talking avatar anything; Claude answers in short sentences, each
tagged with the feeling the avatar's face should show; afterwards a short check asks
about what was said and how the avatar felt. This package is the framework-free core
used by ``legible.api.routers.converse``:

* :mod:`.emotions` — the shared emotion vocabulary and learner-friendly feeling words;
* :mod:`.segments` — the streaming parser that turns tagged text into spoken segments;
* :mod:`.prompts` — Claude's system prompts;
* :mod:`.demo` — curated offline answers when Claude is not configured;
* :mod:`.checks` — comprehension questions (Claude or offline) and scoring;
* :mod:`.tavus` — the optional Tavus CVI client for a realistic face and voice.
"""

from .emotions import EMOTIONS, FEELING_WORDS, NEUTRAL, normalize_emotion
from .segments import Segment, SegmentParser, echo_text, parse_tagged, plain_text, tagged_text

__all__ = [
    "EMOTIONS",
    "FEELING_WORDS",
    "NEUTRAL",
    "Segment",
    "SegmentParser",
    "echo_text",
    "normalize_emotion",
    "parse_tagged",
    "plain_text",
    "tagged_text",
]
