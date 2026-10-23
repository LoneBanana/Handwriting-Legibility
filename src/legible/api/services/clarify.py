""""What was probably meant": turns a literal transcription into the intended text.

The handwriting model reports what is on the page; for a writer with dysgraphia that
often includes misspellings, reversed or missing letters, run-together words, and
letters it could not read confidently. Claude gets the lines with the uncertain letters
marked and the model's alternative readings, and returns a corrected version under a
strict rule: fix form, never add ideas. Every change is listed with a reason so the
writer stays in control of their own words.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from .handwriting import LineResult
from .llm import LLM, LLMError

log = logging.getLogger(__name__)

UNCERTAIN = 0.6

SYSTEM_PROMPT = """\
You help people with dysgraphia (a writing disability) by working out what their handwriting was meant to say.

You receive a machine transcription of a handwritten page, line by line. Letters the handwriting \
model was unsure about are wrapped in [square brackets]; some lines also list alternative readings \
the model considered. The writer may reverse letters (b/d, p/q), drop or double letters, spell \
phonetically, run words together or split them, and skip punctuation.

Return what the writer most likely meant:
- Fix spelling, letter reversals, missing or extra letters, word boundaries, capitalization, and \
punctuation, using the context of the whole page.
- Keep the writer's own words, order, tone, and meaning. Never add facts, ideas, or sentences, \
never remove content, and never "improve" style or vocabulary.
- Keep the line breaks of the original unless two lines are clearly one sentence split mid-word.
- If a word cannot be resolved with reasonable confidence, keep your best guess in the text and \
also list it in `uncertain`.
- In `changes`, list each change as the original fragment, the corrected fragment, and a short, \
kind reason written for the writer (for example "spelling", "b and d swapped", "words joined").
If nothing needs fixing, return the text unchanged with no changes."""


class Change(BaseModel):
    original: str
    corrected: str
    reason: str


class Clarified(BaseModel):
    text: str
    changes: list[Change]
    uncertain: list[str]


def mark_uncertain(line: LineResult, threshold: float = UNCERTAIN) -> str:
    """``hello`` with a doubtful ``e`` becomes ``h[e]llo``; adjacent doubtful letters share brackets."""
    out: list[str] = []
    open_bracket = False
    for score in line.chars:
        doubtful = not score.char.isspace() and score.confidence < threshold
        if doubtful and not open_bracket:
            out.append("[")
            open_bracket = True
        elif not doubtful and open_bracket:
            out.append("]")
            open_bracket = False
        out.append(score.char)
    if open_bracket:
        out.append("]")
    return "".join(out) if line.chars else line.text


def build_prompt(lines: list[LineResult]) -> str:
    parts = ["Transcribed lines:"]
    for line in lines:
        parts.append(f"{line.index + 1}. {mark_uncertain(line)}")
        alternatives = [a for a in line.alternatives if a and a != line.text][:3]
        if alternatives:
            parts.append("   other readings: " + " | ".join(alternatives))
    return "\n".join(parts)


async def clarify(llm: LLM, lines: list[LineResult]) -> Clarified | None:
    """Returns the clarified text, or None when Claude is unavailable or fails (the app
    then simply shows the literal transcription)."""
    if not llm.available or not any(line.text.strip() for line in lines):
        return None
    try:
        return await llm.structured(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(lines)}],
            schema=Clarified,
            max_tokens=4096,
            effort="medium",
        )
    except LLMError as exc:
        log.warning("clarify failed: %s", exc)
        return None
