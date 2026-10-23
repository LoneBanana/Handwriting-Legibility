"""Character and word error rates (the standard HTR metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

try:
    from rapidfuzz.distance import Levenshtein as _lev

    def edit_distance(a: Sequence, b: Sequence) -> int:
        return int(_lev.distance(a, b))

except ImportError:  # pragma: no cover - rapidfuzz is a hard dependency, this is a safety net

    def edit_distance(a: Sequence, b: Sequence) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


@dataclass
class ErrorCounts:
    char_errors: int = 0
    chars: int = 0
    word_errors: int = 0
    words: int = 0
    lines: int = 0

    def add(self, ref: str, hyp: str) -> None:
        self.char_errors += edit_distance(ref, hyp)
        self.chars += len(ref)
        ref_words, hyp_words = ref.split(), hyp.split()
        self.word_errors += edit_distance(ref_words, hyp_words)
        self.words += len(ref_words)
        self.lines += 1

    def merge(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple())))

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.char_errors, self.chars, self.word_errors, self.words, self.lines)

    @property
    def cer(self) -> float:
        return self.char_errors / max(self.chars, 1)

    @property
    def wer(self) -> float:
        return self.word_errors / max(self.words, 1)


def cer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    counts = ErrorCounts()
    for r, h in zip(refs, hyps):
        counts.add(r, h)
    return counts.cer


def wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    counts = ErrorCounts()
    for r, h in zip(refs, hyps):
        counts.add(r, h)
    return counts.wer
