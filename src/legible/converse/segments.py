"""Incremental parsing of Claude's emotion-tagged answers into spoken segments.

Claude starts every sentence with one emotion tag, e.g.
``[excited] Great question! [content] Octopuses have three hearts.``. The avatar speaks
and animates one segment at a time, so the parser hands back each segment the moment
it is complete rather than waiting for the whole answer: a segment ends where the next
tag starts, or at a sentence boundary when Claude left a sentence untagged.

Model output does not always follow the format, so the parser is forgiving:

* a tag may arrive split across stream chunks (``"[exc"`` + ``"ited]"``);
* a sentence without a tag, or with a tag outside the vocabulary, is ``neutral``;
* bracketed text that cannot be a tag (``[1]``, ``[in short]``) stays in the text;
* whitespace, line breaks, markdown emphasis and list markers are tidied away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .emotions import NEUTRAL, normalize_emotion

# A tag is one word in square brackets; the longest "[...]" worth waiting for is
# _MAX_TAG characters, past that an unclosed "[" is just text.
_MAX_TAG = 32
_TAG_BODY = re.compile(r"\s*([A-Za-z][A-Za-z_-]*)\s*")
_TAG_PREFIX = re.compile(r"\[\s*(?:[A-Za-z][A-Za-z_-]*\s*)?")
# End punctuation, any closing quotes or brackets, then whitespace.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’»)\]]*\s+")
_SENTENCE_STARTERS = "\"'“‘(¿¡"
# Words that end in a full stop without ending the sentence ("Dr. Who", "Mt. Everest").
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "st", "mt", "sr", "jr", "vs", "approx", "ft", "lb", "oz"})
_LIST_MARKER = re.compile(r"^(?:[#>•*-]+|\d+[.)])\s+")
_MARKDOWN = re.compile(r"[*`]+")


@dataclass(frozen=True)
class Segment:
    index: int
    emotion: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "emotion": self.emotion, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(index=int(data["index"]), emotion=normalize_emotion(data.get("emotion")), text=str(data["text"]))


class SegmentParser:
    """Feed streamed text with :meth:`feed`; call :meth:`finish` once the stream ends."""

    def __init__(self) -> None:
        self._pending = ""               # received but not yet parsed (may hold a partial tag)
        self._text = ""                  # the segment being built
        self._emotion: str | None = None  # its tag, if it had one
        self._count = 0

    def feed(self, chunk: str) -> list[Segment]:
        self._pending += chunk
        return self._parse(final=False)

    def finish(self) -> list[Segment]:
        segments = self._parse(final=True)
        segments.extend(self._close())
        return segments

    def _parse(self, *, final: bool) -> list[Segment]:
        out: list[Segment] = []
        while self._pending:
            bracket = self._pending.find("[")
            if bracket < 0:
                self._append(self._pending, out)
                self._pending = ""
                break
            if bracket > 0:
                self._append(self._pending[:bracket], out)
                self._pending = self._pending[bracket:]

            close = self._pending.find("]", 1, _MAX_TAG)
            reopen = self._pending.find("[", 1, close if close > 0 else _MAX_TAG)
            if reopen > 0:  # "[[excited]": the first bracket is text
                self._append(self._pending[:reopen], out)
                self._pending = self._pending[reopen:]
                continue
            if close < 0:
                if len(self._pending) < _MAX_TAG and _TAG_PREFIX.fullmatch(self._pending):
                    if final:  # a tag cut off by the end of the answer
                        self._pending = ""
                    break  # otherwise the rest of the tag is still on its way
                self._append("[", out)
                self._pending = self._pending[1:]
                continue

            body = _TAG_BODY.fullmatch(self._pending[1:close])
            if body is None:  # e.g. "[1]"
                self._append(self._pending[: close + 1], out)
            else:
                out.extend(self._close())
                self._emotion = normalize_emotion(body.group(1))
            self._pending = self._pending[close + 1 :]
        return out

    def _append(self, text: str, out: list[Segment]) -> None:
        """Adds text to the current segment, splitting off any sentences it completes."""
        self._text += text
        while (cut := _sentence_end(self._text)) is not None:
            out.extend(self._emit(self._text[:cut]))
            self._text = self._text[cut:]
            self._emotion = None  # the sentence that follows had no tag of its own

    def _close(self) -> list[Segment]:
        segments = self._emit(self._text)
        self._text = ""
        self._emotion = None
        return segments

    def _emit(self, raw: str) -> list[Segment]:
        text = clean_text(raw)
        if not any(ch.isalnum() for ch in text):
            return []
        segment = Segment(index=self._count, emotion=self._emotion or NEUTRAL, text=text)
        self._count += 1
        return [segment]


def _sentence_end(text: str) -> int | None:
    """Index just past the first finished sentence in ``text``, or None if none is certain yet.

    A sentence is finished only once the next one has visibly started (a capital letter,
    digit, or opening quote), which tells "Hmm... let me think" apart from "Hmm... Let's see".
    """
    for match in _SENTENCE_END.finditer(text):
        nxt = match.end()
        if nxt >= len(text):
            return None
        if not (text[nxt].isupper() or text[nxt].isdigit() or text[nxt] in _SENTENCE_STARTERS):
            continue
        if text[match.start()] == "." and _is_abbreviation(text, match.start()):
            continue
        return match.start() + len(match.group().rstrip())
    return None


def _is_abbreviation(text: str, dot: int) -> bool:
    start = dot
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    word = text[start:dot]
    if not word:
        return False
    if word.isdigit():  # "1. First ..." is a list marker, "It was 1969. Then ..." is not
        return not text[:start].strip()
    return len(word) == 1 or "." in word or word.lower() in _ABBREVIATIONS


def clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = _LIST_MARKER.sub("", text)
    return " ".join(_MARKDOWN.sub("", text).split())


def parse_tagged(text: str) -> list[Segment]:
    """Parses a complete tagged answer in one go."""
    parser = SegmentParser()
    return parser.feed(text) + parser.finish()


def plain_text(segments: Iterable[Segment]) -> str:
    return " ".join(segment.text for segment in segments)


def tagged_text(segments: Iterable[Segment]) -> str:
    """The answer in the format Claude writes, for replaying earlier turns as context."""
    return " ".join(f"[{segment.emotion}] {segment.text}" for segment in segments)


def echo_text(segments: Iterable[Segment]) -> str:
    """Text for a Tavus ``conversation.echo``: each sentence preceded by its emotion tag.

    Phoenix-4 faces read ``<emotion value="..."/>`` tags placed before the words they
    apply to. Every segment carries its own tag so a neutral sentence also resets the
    face, and stray ``<`` / ``>`` / ``&`` are spoken as words so they cannot be read as markup.
    """
    return " ".join(f'<emotion value="{segment.emotion}"/> {_speakable(segment.text)}' for segment in segments)


def _speakable(text: str) -> str:
    for symbol, words in (("&", " and "), ("<", " less than "), (">", " greater than ")):
        text = text.replace(symbol, words)
    return " ".join(text.split())
