"""Look-alike and sound-alike choices for "which word did you see?" cards.

Good distractors target the confusions dyslexic readers actually make, so choosing the
right word means really decoding it instead of guessing from the first letter or the
word's shape: mirror-image letters (b/d, p/q), swapped neighbours, a different vowel,
and real words one letter away. Real words from the word bank come first, because a
plausible real word is a stronger test than a made-up one.
"""

from __future__ import annotations

import random
from functools import lru_cache

from ..htr.metrics import edit_distance
from .curriculum import word_bank

VISUAL_SWAPS = {"b": "dp", "d": "bq", "p": "qb", "q": "pd", "m": "nw", "n": "mu", "u": "n", "w": "m",
                "h": "n", "i": "l", "l": "i", "f": "t", "t": "f", "a": "o", "o": "a", "e": "c", "c": "e"}
VOWELS = "aeiou"
# Letter swaps can turn innocent words into crude ones ("duck" with another vowel); never show these.
BLOCKED = frozenset({
    "arse", "ass", "bitch", "boob", "butt", "cock", "crap", "cunt", "damn", "dick", "dyke", "fag", "fart", "fuck",
    "fuk", "hell", "homo", "jerk", "nazi", "piss", "poop", "porn", "puke", "rape", "sex", "shit", "slut", "tit",
    "tits", "turd", "twat", "wank", "whore", "kill", "die", "dead", "gun", "drug", "drunk", "beer", "nude",
})


@lru_cache(maxsize=1)
def _all_words() -> tuple[str, ...]:
    return tuple(sorted({w.text for words in word_bank().values() for w in words}))


def _variants(word: str) -> list[str]:
    """Generated near-misses: mirror letters, swapped neighbours, other vowels, and the
    word backwards when that is itself a word (was/saw, dog/god)."""
    out: list[str] = []
    for i, ch in enumerate(word):
        for swap in VISUAL_SWAPS.get(ch, ""):
            out.append(word[:i] + swap + word[i + 1 :])
    for i in range(len(word) - 1):
        if word[i] != word[i + 1]:
            out.append(word[:i] + word[i + 1] + word[i] + word[i + 2 :])
    for i, ch in enumerate(word):
        if ch in VOWELS:
            out.extend(word[:i] + v + word[i + 1 :] for v in VOWELS if v != ch)
    if word[::-1] in _all_words():
        out.append(word[::-1])
    return out


def distractors(word: str, count: int, seed: str) -> list[str]:
    """``count`` distinct confusable alternatives to ``word`` (never ``word`` itself)."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    lowered = word.lower()
    real = [w for w in _all_words() if w != lowered and abs(len(w) - len(lowered)) <= 1
            and edit_distance(w, lowered) <= (1 if len(lowered) <= 4 else 2)]
    rng.shuffle(real)
    generated = [v for v in dict.fromkeys(_variants(lowered)) if v != lowered and v.isalpha() and v not in BLOCKED]
    rng.shuffle(generated)
    picks: list[str] = []
    # At most half real words, so each card also tests letter-level decoding.
    for candidate in real[: max(1, count // 2)] + generated + real[max(1, count // 2):]:
        if candidate not in picks and candidate != lowered:
            picks.append(candidate)
        if len(picks) == count:
            break
    return picks
