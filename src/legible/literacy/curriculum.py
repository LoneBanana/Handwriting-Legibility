"""The structured-literacy scope and sequence, and the word bank behind it.

Skills follow the order structured-literacy programs use (in the Orton-Gillingham
tradition): short vowels in closed syllables, consonant digraphs, blends, doubling
rules, vowel-consonant-e, vowel teams, r-controlled vowels, diphthongs, soft c/g,
trigraphs and silent letters, then endings, prefixes, -tion/-sion and two-syllable
words. Each skill is practised until mastered before the next is introduced, and
earlier skills keep returning in spaced review.

Words live in ``data/*.tsv`` (``skill  word  graphemes  sentence``), with graphemes
separated by dots: ``sh.i.p``, ``c.a.k.e``, ``n.igh.t``. The app colours each grapheme
and highlights the skill's pattern (its *focus* span) inside the word.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

LEVELS_PER_SKILL = 8


@dataclass(frozen=True)
class Skill:
    id: str
    label: str                                  # learner-facing, e.g. "ph says /f/"
    patterns: tuple[str, ...] = ()              # letter strings that make up the pattern
    position: Literal["any", "start", "end"] = "any"
    vce: str | None = None                      # vowel of a vowel-consonant-e skill


@dataclass(frozen=True)
class Word:
    skill: str
    text: str
    graphemes: tuple[str, ...]
    sentence: str

    @property
    def focus(self) -> tuple[int, int] | None:
        return focus_span(self.text, self.graphemes, SKILLS_BY_ID[self.skill])


SKILLS: tuple[Skill, ...] = (
    Skill("short_a", "short a, as in cat", ("a",)),
    Skill("short_i", "short i, as in pig", ("i",)),
    Skill("short_o", "short o, as in dog", ("o",)),
    Skill("short_u", "short u, as in sun", ("u",)),
    Skill("short_e", "short e, as in bed", ("e",)),
    Skill("digraph_sh", "sh, as in ship", ("sh",)),
    Skill("digraph_ch", "ch, as in chip", ("ch",)),
    Skill("digraph_th", "th, as in thin", ("th",)),
    Skill("digraph_wh", "wh, as in when", ("wh",), position="start"),
    Skill("ck", "ck after a short vowel, as in duck", ("ck",)),
    Skill("ng_nk", "ng and nk, as in ring and pink", ("ng", "nk")),
    Skill("blends_l", "l-blends: bl, cl, fl, gl, pl, sl", ("bl", "cl", "fl", "gl", "pl", "sl"), position="start"),
    Skill("blends_r", "r-blends: br, cr, dr, fr, gr, pr, tr", ("br", "cr", "dr", "fr", "gr", "pr", "tr"),
          position="start"),
    Skill("blends_s", "s-blends: sc, sk, sm, sn, sp, st, sw", ("sc", "sk", "sm", "sn", "sp", "st", "sw"),
          position="start"),
    Skill("end_blends", "end blends: nd, nt, mp, st, ft, lt", ("nd", "nt", "mp", "st", "ft", "lt"), position="end"),
    Skill("floss", "ff, ll, ss after a short vowel", ("ff", "ll", "ss"), position="end"),
    Skill("vce_a", "a-consonant-e, as in cake", vce="a"),
    Skill("vce_i", "i-consonant-e, as in bike", vce="i"),
    Skill("vce_o", "o-consonant-e, as in home", vce="o"),
    Skill("vce_u", "u-consonant-e, as in cute", vce="u"),
    Skill("ai_ay", "ai and ay, as in rain and play", ("ai", "ay")),
    Skill("ee_ea", "ee and ea, as in tree and sea", ("ee", "ea")),
    Skill("oa_ow", "oa and ow, as in boat and snow", ("oa", "ow")),
    Skill("igh_y", "igh and y, as in night and fly", ("igh", "y")),
    Skill("ar", "ar, as in car", ("ar",)),
    Skill("or", "or, as in fork", ("or",)),
    Skill("er_ir_ur", "er, ir and ur, as in her, bird and fur", ("er", "ir", "ur")),
    Skill("oi_oy", "oi and oy, as in coin and boy", ("oi", "oy")),
    Skill("ou_ow", "ou and ow, as in cloud and cow", ("ou", "ow")),
    Skill("oo", "oo, as in moon and book", ("oo",)),
    Skill("soft_c_g", "soft c and g before e, i or y", ("ce", "ci", "cy", "ge", "gi", "gy")),
    Skill("tch_dge", "tch and dge after a short vowel", ("tch", "dge")),
    Skill("silent_letters", "ph says /f/; kn and wr have a silent letter", ("ph", "kn", "wr")),
    Skill("ending_ing_ed", "endings -ing and -ed", ("ing", "ed"), position="end"),
    Skill("plural_s_es", "plurals -s and -es", ("es", "s"), position="end"),
    Skill("prefix_un_re", "prefixes un- and re-", ("un", "re"), position="start"),
    Skill("tion_sion", "-tion and -sion say /shun/", ("tion", "sion"), position="end"),
    Skill("two_syllable", "two-syllable words, split between consonants"),
)

SKILLS_BY_ID: dict[str, Skill] = {skill.id: skill for skill in SKILLS}


def skill_index_for(reading_level: int, spelling_level: int) -> int:
    """New patterns arrive only when both reading and spelling have mastered the current one."""
    return min(min(reading_level, spelling_level) // LEVELS_PER_SKILL, len(SKILLS) - 1)


def _boundaries(graphemes: tuple[str, ...]) -> list[int]:
    edges = [0]
    for g in graphemes:
        edges.append(edges[-1] + len(g))
    return edges


def focus_span(word: str, graphemes: tuple[str, ...], skill: Skill) -> tuple[int, int] | None:
    """The [start, end) characters of the skill's pattern in ``word``, aligned to graphemes."""
    lower = word.lower()
    edges = _boundaries(graphemes)
    if skill.vce:
        # vowel, one consonant grapheme, final silent e: c-a-k-e -> "ake"
        if len(graphemes) >= 3 and graphemes[-1] == "e" and graphemes[-3] == skill.vce:
            return edges[-4], edges[-1]
        return None
    starts = set(edges)
    for pattern in skill.patterns:
        if skill.position == "start":
            candidates = [0] if lower.startswith(pattern) else []
        elif skill.position == "end":
            candidates = [len(lower) - len(pattern)] if lower.endswith(pattern) else []
        else:
            candidates = [i for i in range(len(lower)) if lower.startswith(pattern, i)]
        for start in candidates:
            end = start + len(pattern)
            if start in starts and end in starts:
                return start, end
    return None


@lru_cache(maxsize=1)
def word_bank() -> dict[str, tuple[Word, ...]]:
    """All words by skill id, in curriculum order (loaded once from package data)."""
    words: dict[str, list[Word]] = {skill.id: [] for skill in SKILLS}
    folder = resources.files("legible.literacy") / "data"
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".tsv"):
            continue
        with entry.open(encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if not row or row[0].startswith("#"):
                    continue
                skill, text, graphemes, sentence = (cell.strip() for cell in row)
                if skill not in words:
                    raise ValueError(f"{entry.name}: unknown skill {skill!r}")
                words[skill].append(Word(skill, text, tuple(graphemes.split(".")), sentence))
    return {skill: tuple(items) for skill, items in words.items()}


def find_word(text: str) -> Word | None:
    lowered = text.lower()
    for items in word_bank().values():
        for word in items:
            if word.text == lowered:
                return word
    return None
