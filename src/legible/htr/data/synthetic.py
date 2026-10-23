"""Synthetic handwriting with the irregularities typical of dysgraphia.

Public handwriting datasets are mostly neat adult writing, and there is no large
transcribed corpus of dysgraphic writing. So lines are rendered here letter by letter
from handwriting fonts with per-letter jitter that follows the traits handwriting
assessments score (e.g. the BHK scale): inconsistent letter size, letters floating off
the baseline, variable slant, cramped or spread letter spacing, word gaps that are too
small or too large, mirrored letters (b/d, p/q), and crossed-out false starts. The
label is always the *intended* text, so the model learns to read through those traits.
Stroke-level wobble, baseline drift and photo effects are added afterwards by
:mod:`.augment`.

``irregularity`` (0..1) scales every jitter: 0 renders neat font text, 1 is strongly
irregular writing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}

# Handwriting-like fonts that ship with common operating systems, used when no
# fonts directory is configured (see scripts/download_fonts.py for a larger set).
SYSTEM_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Noteworthy.ttc",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/MarkerFelt.ttc",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/usr/share/fonts/truetype/tlwg/Purisa.ttf",
    "/usr/share/fonts/truetype/comic-neue/ComicNeue-Regular.ttf",
    "C:/Windows/Fonts/segoepr.ttf",
    "C:/Windows/Fonts/comic.ttf",
]

# Letters commonly written mirrored; the label keeps the intended letter.
_MIRRORABLE = set("bdpqsjzJSZ2345679")


@dataclass(frozen=True)
class FontSpec:
    path: str | None                 # None = Pillow's bundled default font
    chars: frozenset[str]

    @property
    def name(self) -> str:
        return Path(self.path).stem if self.path else "default"


def _font_chars(path: str) -> frozenset[str]:
    from fontTools.ttLib import TTFont

    logging.getLogger("fontTools").setLevel(logging.ERROR)  # harmless table warnings in system fonts
    font = TTFont(path, fontNumber=0, lazy=True)
    try:
        cmap = font.getBestCmap() or {}
    finally:
        font.close()
    return frozenset(chr(cp) for cp in cmap)


def discover_fonts(fonts_dir: str | Path | None = None) -> list[FontSpec]:
    """Fonts from ``fonts_dir`` (recursive), else system handwriting fonts, else Pillow's default."""
    if fonts_dir:
        paths = sorted(p for p in Path(fonts_dir).rglob("*") if p.suffix.lower() in FONT_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no .ttf/.otf/.ttc fonts found under {fonts_dir}")
    else:
        paths = [Path(p) for p in SYSTEM_FONT_CANDIDATES if Path(p).exists()]
    fonts = []
    for path in paths:
        try:
            chars = _font_chars(str(path))
            ImageFont.truetype(str(path), 32, index=0)
        except Exception as exc:  # noqa: BLE001 - skip unreadable fonts, keep the rest
            log.warning("skipping font %s: %s", path, exc)
            continue
        if sum(c.isalpha() for c in chars) >= 52:
            fonts.append(FontSpec(str(path), chars))
    if not fonts:
        log.warning("no handwriting fonts found; synthetic data falls back to Pillow's default font")
        fonts.append(FontSpec(None, frozenset(chr(c) for c in range(32, 127))))
    return fonts


@lru_cache(maxsize=512)
def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    if path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(path, size, index=0)


@lru_cache(maxsize=8192)
def _glyph(path: str | None, size: int, text: str, mirror: bool) -> tuple[np.ndarray, int, int, float]:
    """Renders ``text`` once: (uint8 ink image, x offset, y offset from the baseline origin, advance)."""
    font = _font(path, size)
    x0, y0, x1, y1 = font.getbbox(text, anchor="ls")
    pad = 3
    w, h = max(x1 - x0, 1) + 2 * pad, max(y1 - y0, 1) + 2 * pad
    image = Image.new("L", (w, h), 0)
    ImageDraw.Draw(image).text((pad - x0, pad - y0), text, font=font, fill=255, anchor="ls")
    array = np.asarray(image, dtype=np.uint8)
    if mirror:
        array = array[:, ::-1]
    return array, x0 - pad, y0 - pad, float(font.getlength(text))


def _rotate(glyph: np.ndarray, ox: int, oy: int, degrees: float) -> tuple[np.ndarray, int, int]:
    """Rotates a glyph about its centre, returning the new image and baseline offsets."""
    if abs(degrees) < 0.2:
        return glyph, ox, oy
    h, w = glyph.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(glyph, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    return out, int(round(ox + w / 2 - nw / 2)), int(round(oy + h / 2 - nh / 2))


class _Canvas:
    def __init__(self, width: int, height: int, baseline: int):
        self.ink = np.zeros((height, width), np.uint8)
        self.baseline = baseline

    def paste(self, glyph: np.ndarray, x: float, ox: int, oy: int, dy: float = 0.0) -> None:
        h, w = glyph.shape
        left, top = int(round(x)) + ox, self.baseline + oy + int(round(dy))
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + w, self.ink.shape[1]), min(top + h, self.ink.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        region = self.ink[y0:y1, x0:x1]
        np.maximum(region, glyph[y0 - top : y1 - top, x0 - left : x1 - left], out=region)

    def scribble(self, x0: float, x1: float, size: int, rng: np.random.Generator) -> None:
        """Crosses out [x0, x1]: a zig-zag or a few strike-through lines."""
        top = self.baseline - int(size * 0.75)
        bottom = self.baseline - int(size * 0.05)
        thickness = max(2, size // 14)
        if rng.random() < 0.5:
            xs = np.linspace(x0, x1, int(rng.integers(4, 9)))
            ys = np.where(np.arange(len(xs)) % 2 == 0, top, bottom) + rng.normal(0, size * 0.05, len(xs))
            points = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(self.ink, [points], False, 255, thickness, cv2.LINE_AA)
        else:
            for _ in range(int(rng.integers(1, 4))):
                y = int(rng.uniform(top + size * 0.15, bottom))
                tilt = int(rng.normal(0, size * 0.08))
                cv2.line(self.ink, (int(x0), y + tilt), (int(x1), y - tilt), 255, thickness, cv2.LINE_AA)


class HandwritingRenderer:
    def __init__(self, fonts: list[FontSpec], irregularity: float = 0.5, size: int = 56,
                 p_mirror: float = 0.03, p_strike: float = 0.04):
        if not fonts:
            raise ValueError("at least one font is required")
        self.fonts = fonts
        self.irregularity = float(np.clip(irregularity, 0.0, 1.0))
        self.size = size
        self.p_mirror = p_mirror
        self.p_strike = p_strike

    def render(self, text: str, rng: np.random.Generator,
               irregularity: float | None = None) -> tuple[np.ndarray, str]:
        """Returns (float32 ink map with ink=1, label). Characters the chosen font
        cannot draw are dropped from both the image and the label."""
        font = self.fonts[int(rng.integers(len(self.fonts)))]
        words = ["".join(c for c in w if c in font.chars) for w in text.split(" ")]
        words = [w for w in words if w]
        label = " ".join(words)
        if not words:
            return np.zeros((self.size, self.size), np.float32), ""

        # Writers vary; capping at 1 keeps every line readable by a patient human, so
        # the model is never trained to hallucinate text nobody could read.
        base = self.irregularity if irregularity is None else float(np.clip(irregularity, 0.0, 1.0))
        irr = min(1.0, base * rng.uniform(0.5, 1.2))
        size = int(round(self.size * math.exp(rng.normal(0, 0.08))))
        space = _font(font.path, size).getlength(" ") or size * 0.3
        spacing = math.exp(rng.normal(0, 0.12))  # this writer's overall letter spacing
        width = int(sum(len(w) + 1 for w in words) * size * 1.6 + 4 * size)
        canvas = _Canvas(width, int(size * 3.2), baseline=int(size * 2.0))
        x = float(size)

        for i, word in enumerate(words):
            if i:
                gap = math.exp(rng.normal(0, 0.45 * irr)) * (0.6 if rng.random() < 0.15 * irr else 1.0)
                x += space * max(gap, 0.3)
            if rng.random() < self.p_strike * irr:
                x = self._false_start(canvas, font, size, x, rng)
            if rng.random() < 0.6 * (1.0 - irr):
                x = self._word(canvas, font, size, word, x, irr, rng)
            else:
                x = self._letters(canvas, font, size, word, x, irr, spacing, rng)

        ink = canvas.ink[:, : int(min(width, x + size))]
        rows = np.flatnonzero(ink.max(axis=1) > 32)
        cols = np.flatnonzero(ink.max(axis=0) > 32)
        if rows.size == 0:
            return np.zeros((self.size, self.size), np.float32), ""
        margin = max(2, size // 10)
        y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, ink.shape[0])
        x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, ink.shape[1])
        return ink[y0:y1, x0:x1].astype(np.float32) / 255.0, label

    def _word(self, canvas: _Canvas, font: FontSpec, size: int, word: str, x: float, irr: float,
              rng: np.random.Generator) -> float:
        """Whole word in one pass, keeping a cursive font's joins."""
        word_size = max(8, int(round(size * math.exp(rng.normal(0, 0.1 * irr)))))
        glyph, ox, oy, advance = _glyph(font.path, word_size, word, False)
        glyph, ox, oy = _rotate(glyph, ox, oy, rng.normal(0, 3.0 * irr))
        canvas.paste(glyph, x, ox, oy, dy=rng.normal(0, 0.08 * irr) * size)
        return x + advance

    def _letters(self, canvas: _Canvas, font: FontSpec, size: int, word: str, x: float, irr: float,
                 spacing: float, rng: np.random.Generator) -> float:
        """Letter by letter, each with its own size, height above the baseline, and slant."""
        for char in word:
            scale = math.exp(rng.normal(0, 0.18 * irr))
            if char.islower() and rng.random() < 0.05 * irr:
                scale *= rng.uniform(1.3, 1.7)  # a small letter written as tall as a capital
            letter_size = max(8, int(round(size * scale)))
            mirror = char in _MIRRORABLE and rng.random() < self.p_mirror * irr
            glyph, ox, oy, advance = _glyph(font.path, letter_size, char, mirror)
            glyph, ox, oy = _rotate(glyph, ox, oy, rng.normal(0, 9.0 * irr))
            canvas.paste(glyph, x, ox, oy, dy=rng.normal(0, 0.12 * irr) * size)
            x += advance * spacing * max(math.exp(rng.normal(0, 0.2 * irr)), 0.75)
        return x

    def _false_start(self, canvas: _Canvas, font: FontSpec, size: int, x: float, rng: np.random.Generator) -> float:
        """A wrong letter or two, crossed out, before the real word (not part of the label)."""
        letters = [c for c in "abcdefghijklmnopqrstuvwxyz" if c in font.chars]
        if not letters:
            return x
        start = x
        for _ in range(int(rng.integers(1, 3))):
            glyph, ox, oy, advance = _glyph(font.path, size, str(rng.choice(letters)), False)
            canvas.paste(glyph, x, ox, oy)
            x += advance
        canvas.scribble(start - size * 0.05, x + size * 0.05, size, rng)
        return x + size * 0.25
