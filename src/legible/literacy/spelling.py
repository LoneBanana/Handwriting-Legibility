"""Explains spelling mistakes the way a reading specialist would.

The learner's answer is aligned with the target word (Levenshtein with a backtrace) and
each difference is named: a **reversal** (b/d, p/q, m/w, n/u, letters that are mirror
images or rotations of each other), a **transposition** (two neighbours swapped), a
missing or extra **double** letter, a **vowel** spelled with another vowel, or a plain
omission/insertion/substitution. Naming the error type matters: "b and d are mirror
images" teaches something; "wrong" does not.

``phonetically_plausible`` asks whether the answer *sounds* like the word ("fone" for
"phone", "nite" for "night"): a good sign that the learner hears the sounds and only
needs the spelling pattern, which deserves different feedback from a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ErrorType = Literal["reversal", "transposition", "omission", "insertion", "substitution", "vowel", "doubling"]

VOWELS = set("aeiou")
REVERSALS = {("b", "d"), ("d", "b"), ("p", "q"), ("q", "p"), ("m", "w"), ("w", "m"), ("n", "u"), ("u", "n")}


@dataclass(frozen=True)
class SpellingError:
    type: ErrorType
    position: int          # index in the target word
    detail: str

    def as_dict(self) -> dict:
        return {"type": self.type, "position": self.position, "detail": self.detail}


def normalize_answer(text: str) -> str:
    """Lower-case letters (and apostrophes) only, so "Cat." and " cat " both match "cat"."""
    return re.sub(r"[^a-z']", "", text.strip().lower())


def _align(expected: str, response: str) -> list[tuple[str, int, int]]:
    """Edit operations turning ``expected`` into ``response``: (op, i, j) with op in
    match/sub/del/ins, i indexing ``expected`` and j indexing ``response``."""
    n, m = len(expected), len(response)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = expected[i - 1] == response[j - 1]
            cost[i][j] = min(cost[i - 1][j - 1] + (0 if same else 1), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (0 if expected[i - 1] == response[j - 1] else 1):
            ops.append(("match" if expected[i - 1] == response[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    ops.reverse()
    return ops


def analyze(expected: str, response: str, chosen: bool = False) -> list[SpellingError]:
    """Named differences between the target word and the learner's spelling (or, with
    ``chosen``, the look-alike word they picked)."""
    wrote = "Your choice has" if chosen else "You wrote"
    expected, response = expected.lower(), normalize_answer(response)
    ops = _align(expected, response)
    errors: list[SpellingError] = []
    k = 0
    while k < len(ops):
        op, i, j = ops[k]
        nxt = ops[k + 1] if k + 1 < len(ops) else None
        if op == "match":
            k += 1
            continue
        # Two neighbouring letters swapped: "form" -> "from", "was" -> "saw" (middle)
        if (op == "sub" and nxt and nxt[0] == "sub" and nxt[1] == i + 1 and nxt[2] == j + 1
                and expected[i] == response[j + 1] and expected[i + 1] == response[j]):
            pair, swapped = expected[i : i + 2], response[j : j + 2]
            errors.append(SpellingError("transposition", i, f"“{pair}” came out as “{swapped}”: the two letters swapped places."))
            k += 2
            continue
        if op == "sub":
            want, got = expected[i], response[j]
            if (got, want) in REVERSALS:
                errors.append(SpellingError("reversal", i, f"{wrote} “{got}” where the word has “{want}”. "
                                                         f"They look like mirror images, so they are easy to mix up."))
            elif want in VOWELS and got in VOWELS:
                errors.append(SpellingError("vowel", i, f"The vowel here is “{want}”, not “{got}”."))
            else:
                errors.append(SpellingError("substitution", i, f"{wrote} “{got}” where the word has “{want}”."))
        elif op == "del":
            want = expected[i]
            doubled = (i > 0 and expected[i - 1] == want) or (i + 1 < len(expected) and expected[i + 1] == want)
            if doubled:
                errors.append(SpellingError("doubling", i, f"This word has a double “{want}{want}”."))
            else:
                errors.append(SpellingError("omission", i, f"The “{want}” is missing."))
        else:  # ins
            got = response[j]
            near = expected[i - 1] if i > 0 else ""
            after = expected[i] if i < len(expected) else ""
            if got in (near, after):
                errors.append(SpellingError("doubling", min(i, max(len(expected) - 1, 0)),
                                            f"This word has just one “{got}” here."))
            else:
                errors.append(SpellingError("insertion", min(i, max(len(expected) - 1, 0)),
                                            f"There is an extra “{got}”."))
        k += 1
    return errors


# ---------------------------------------------------------------- sounds alike

_RULES: list[tuple[str, str]] = [
    (r"tch", "C"), (r"dge", "J"), (r"tion$", "Sn"), (r"sion$", "Sn"), (r"igh", "I"),
    (r"ch", "C"), (r"sh", "S"), (r"th", "T"), (r"ph", "f"), (r"wh", "w"), (r"ck", "k"),
    (r"^kn", "n"), (r"^wr", "r"), (r"qu", "kw"), (r"x", "ks"),
    (r"ai|ay|eigh", "A"), (r"ee|ea|ie", "E"), (r"oa|ow$|oe", "O"), (r"oi|oy", "Y"),
    (r"ou|ow", "W"), (r"oo|ew|ue", "U"), (r"ar", "R"), (r"or|ore", "Q"), (r"er|ir|ur", "Z"),
    (r"c(?=[eiy])", "s"), (r"g(?=[eiy])", "j"), (r"c", "k"), (r"q", "k"),
]
_MAGIC_E = re.compile(r"([aeiou])([^aeiouAEIOUYWQRZ])e$")
_LONG = {"a": "A", "e": "E", "i": "I", "o": "O", "u": "U"}
_CONSONANTS = set("bcdfghjklmnpqrstvwxz")


def sound_key(word: str) -> str:
    """A rough phonetic spelling: words that sound the same map to the same key."""
    w = normalize_answer(word).replace("'", "")
    if not w:
        return ""
    if len(w) > 3 and w.endswith("ed"):
        w = w[:-2] + "d"                                       # jumped ~ jumpd
    w = _MAGIC_E.sub(lambda m: _LONG[m.group(1)] + m.group(2), w)  # cake -> kAk
    for pattern, sound in _RULES:
        w = re.sub(pattern, sound, w)
    if len(w) > 2 and w[-1] == "t" and w[-2] in _CONSONANTS:
        w = w[:-1] + "d"                                       # jumpt ~ jumped
    w = re.sub(r"y$", "I" if len(w) <= 3 else "E", w)          # my, fly / happy
    w = re.sub(r"(?<=[^aeiou])e$", "", w)                      # silent final e
    return re.sub(r"(.)\1+", r"\1", w)                         # bell -> bel


def phonetically_plausible(expected: str, response: str) -> bool:
    answer = normalize_answer(response)
    return bool(answer) and answer != expected.lower() and sound_key(answer) == sound_key(expected)
