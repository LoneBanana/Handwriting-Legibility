"""Label-preserving augmentations that mimic dysgraphic handwriting.

Dysgraphia shows up in images as inconsistent letter size and slant, drifting or wavy
baselines, irregular letter/word spacing, shaky (tremor-like) strokes, uneven pen
pressure, and broken or overwritten strokes. Each transform below targets one of those
traits; ``strength`` scales the geometric ones so a curriculum can ramp them up.

All functions take and return float32 ink maps (ink=1, paper=0) and draw randomness
only from the ``rng`` they are given, so samples are reproducible per index.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import AugmentConfig

_INTERP = cv2.INTER_LINEAR
_BORDER = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def random_affine(ink: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Rotation, slant (shear), and letter-width/height scaling, without cropping."""
    h, w = ink.shape
    angle = math.radians(rng.uniform(-3.0, 3.0) * strength)
    shear = rng.uniform(-0.45, 0.45) * strength
    sx = math.exp(rng.uniform(-0.3, 0.3) * strength)
    sy = math.exp(rng.uniform(-0.12, 0.12) * strength)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    a = rot @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([sx, sy])
    center = np.array([w / 2.0, h / 2.0])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64) - center
    moved = corners @ a.T
    lo, hi = moved.min(0), moved.max(0)
    out_w, out_h = int(math.ceil(hi[0] - lo[0])), int(math.ceil(hi[1] - lo[1]))
    m = np.hstack([a, (-(a @ center) - lo)[:, None]])
    return cv2.warpAffine(ink, m, (max(out_w, 1), max(out_h, 1)), flags=_INTERP, **_BORDER)


def horizontal_jitter(ink: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Non-uniform horizontal stretching: cramped letters next to spread-out ones."""
    h, w = ink.shape
    n = max(2, int(w / (h * rng.uniform(0.4, 1.2))))
    scales = np.exp(rng.normal(0.0, 0.3 * strength, n))
    x_in = np.linspace(0.0, w, n + 1)
    x_out = np.concatenate([[0.0], np.cumsum(scales * (w / n))])
    new_w = max(int(round(x_out[-1])), 1)
    map_x = np.interp(np.arange(new_w, dtype=np.float64) + 0.5, x_out, x_in).astype(np.float32) - 0.5
    map_x = np.repeat(map_x[None, :], h, axis=0)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], new_w, axis=1)
    return cv2.remap(ink, map_x, map_y, _INTERP, **_BORDER)


def baseline_warp(ink: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Wavy, drifting baseline: each column is shifted vertically by a smooth curve."""
    h, w = ink.shape
    x = np.arange(w, dtype=np.float32)
    amp = rng.uniform(0.03, 0.14) * h * strength
    l1, l2 = rng.uniform(0.4, 2.0) * w, rng.uniform(0.1, 0.5) * w
    dy = amp * (np.sin(2 * np.pi * x / l1 + rng.uniform(0, 2 * np.pi))
                + 0.4 * np.sin(2 * np.pi * x / l2 + rng.uniform(0, 2 * np.pi)))
    dy += rng.uniform(-0.15, 0.15) * strength * h * (x - w / 2) / max(w, 1)  # overall drift up/down
    pad = int(math.ceil(np.abs(dy).max())) + 1
    canvas = cv2.copyMakeBorder(ink, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0.0)
    map_x = np.repeat(x[None, :], h + 2 * pad, axis=0)
    map_y = np.arange(h + 2 * pad, dtype=np.float32)[:, None] - dy[None, :]
    return cv2.remap(canvas, map_x, map_y.astype(np.float32), _INTERP, **_BORDER)


def elastic(ink: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Smooth random displacement field: shaky, tremor-like or wobbly strokes."""
    h, w = ink.shape
    tremor = rng.random() < 0.5
    sigma = (rng.uniform(0.15, 0.3) if tremor else rng.uniform(0.6, 1.2)) * h / 8
    alpha = (rng.uniform(0.3, 0.8) if tremor else rng.uniform(0.8, 2.0)) * strength * h / 32
    fields = []
    for _ in range(2):
        f = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        fields.append(f / (f.std() + 1e-6) * alpha)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(ink, xx + fields[0], yy + fields[1], _INTERP, **_BORDER)


def stroke_width(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavier or lighter pen pressure."""
    k = int(rng.integers(2, 4))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if rng.random() < 0.65:
        return cv2.dilate(ink, kernel)
    return cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))


def ink_fade(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uneven pressure / drying pen: ink intensity varies smoothly along the line."""
    h, w = ink.shape
    coarse = rng.uniform(0.35, 1.0, (max(h // 16, 2), max(w // 16, 2))).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return ink * np.clip(field, 0.3, 1.0)


def cutout(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small erased patches: broken strokes and pen skips."""
    out = ink.copy()
    h, w = ink.shape
    for _ in range(int(rng.integers(1, 4))):
        ch, cw = int(rng.uniform(0.08, 0.35) * h), int(rng.uniform(0.04, 0.2) * h)
        y, x = int(rng.integers(0, max(h - ch, 1))), int(rng.integers(0, max(w - cw, 1)))
        out[y:y + ch, x:x + cw] = 0.0
    return out


def paper(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Notebook paper: ruled lines, an optional margin line, and fiber texture."""
    h, w = ink.shape
    bg = np.zeros_like(ink)
    if rng.random() < 0.7:
        for frac in rng.choice([0.85, 0.15, 0.5], size=int(rng.integers(1, 3)), replace=False):
            y = int(frac * h + rng.normal(0, 0.04 * h))
            slope = rng.normal(0, 0.01)
            pts = np.array([[0, y], [w - 1, int(y + slope * w)]], dtype=np.int32)
            cv2.polylines(bg, [pts], False, float(rng.uniform(0.12, 0.35)), int(rng.integers(1, 3)))
    if rng.random() < 0.15:
        x = int(rng.uniform(0.02, 0.2) * w)
        cv2.line(bg, (x, 0), (x, h - 1), float(rng.uniform(0.1, 0.3)), 1)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(0.5, 2.0))
    bg += np.abs(texture) * rng.uniform(0.0, 0.05)
    return np.maximum(ink, bg)


def blur(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h = ink.shape[0]
    if rng.random() < 0.3:  # camera shake / motion blur
        k = max(3, int(rng.uniform(0.02, 0.06) * h) | 1)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        return cv2.filter2D(ink, -1, kernel / max(kernel.sum(), 1e-6))
    return cv2.GaussianBlur(ink, (0, 0), rng.uniform(0.3, 1.0) * h / 64)


def noise(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return ink + rng.normal(0, rng.uniform(0.01, 0.07), ink.shape).astype(np.float32)


def jpeg(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = (255.0 * (1.0 - np.clip(ink, 0, 1))).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 85))])
    if not ok:
        return ink
    return 1.0 - cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


class Augmenter:
    """Applies the configured transforms in a fixed, physically sensible order:
    writing geometry first, then pen/ink effects, then paper and camera effects."""

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c, s = self.cfg, self.cfg.strength
        if not c.enabled:
            return ink
        steps = (
            (c.p_affine, lambda x: random_affine(x, rng, s)),
            (c.p_horizontal_jitter, lambda x: horizontal_jitter(x, rng, s)),
            (c.p_baseline_warp, lambda x: baseline_warp(x, rng, s)),
            (c.p_elastic, lambda x: elastic(x, rng, s)),
            (c.p_stroke_width, lambda x: stroke_width(x, rng)),
            (c.p_ink_fade, lambda x: ink_fade(x, rng)),
            (c.p_cutout, lambda x: cutout(x, rng)),
            (c.p_paper, lambda x: paper(x, rng)),
            (c.p_blur, lambda x: blur(x, rng)),
            (c.p_noise, lambda x: noise(x, rng)),
            (c.p_jpeg, lambda x: jpeg(x, rng)),
        )
        for p, fn in steps:
            if rng.random() < p:
                ink = fn(ink)
        return np.clip(ink, 0.0, 1.0).astype(np.float32)
