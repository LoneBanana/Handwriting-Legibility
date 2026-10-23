"""Character vocabulary shared by the CTC head and the attention decoder.

Index 0 doubles as the CTC blank and the decoder padding token: the decoder never
predicts padding and CTC never uses padding as a target, so sharing the slot keeps a
single output layer size for both heads.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)

# Printable ASCII, common Latin-1 letters for names and loanwords, a few symbols that
# show up in school work, and newline for paragraph-level decoding.
DEFAULT_CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
    "£€°±×÷§"
    "àâäáçéèêëíîïñóôöùúûüÿæœß"
    "ÀÂÄÁÇÉÈÊËÍÎÏÑÓÔÖÙÚÛÜŸÆŒ"
    "\n"
)

# Typographic variants that are visually indistinguishable in handwriting are folded
# onto one canonical character so the model never has to guess between them.
_FOLD = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
        "“": '"', "”": '"', "„": '"', "″": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
        " ": " ", "\t": " ", "\r": "",
    }
)


def normalize_text(text: str) -> str:
    """Canonicalize a transcription: Unicode NFKC, folded punctuation, collapsed spaces."""
    text = unicodedata.normalize("NFKC", text).translate(_FOLD)
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(lines).strip()


class Vocab:
    """Bidirectional mapping between characters and integer ids."""

    pad_id = 0
    blank_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    def __init__(self, chars: Iterable[str] = DEFAULT_CHARSET):
        unique = list(dict.fromkeys(chars))
        bad = [c for c in unique if len(c) != 1 or c in SPECIAL_TOKENS]
        if bad:
            raise ValueError(f"vocabulary entries must be single characters, got {bad!r}")
        self.itos: list[str] = list(SPECIAL_TOKENS) + unique
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, char: str) -> bool:
        return char in self.stoi and self.stoi[char] >= len(SPECIAL_TOKENS)

    @property
    def chars(self) -> str:
        return "".join(self.itos[len(SPECIAL_TOKENS):])

    def encode(self, text: str, normalize: bool = True) -> list[int]:
        if normalize:
            text = normalize_text(text)
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id and stop_at_eos:
                break
            if i < len(SPECIAL_TOKENS):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def coverage(self, text: str) -> float:
        """Fraction of characters in ``text`` that the vocabulary can represent."""
        text = normalize_text(text)
        if not text:
            return 1.0
        return sum(c in self.stoi for c in text) / len(text)

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars}, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "Vocab":
        return cls(json.loads(payload)["chars"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
