"""Image I/O and normalization into the model's input space.

The model sees an "ink map": float32 in [0, 1] with ink = 1 and paper = 0, so right
padding with zeros is indistinguishable from blank paper.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_gray(source: str | Path | bytes | Image.Image | np.ndarray) -> np.ndarray:
    """Load any image source as uint8 grayscale (H, W), honoring EXIF rotation."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY if arr.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return arr
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))
    return np.asarray(img.convert("L"), dtype=np.uint8)


def flatten_background(gray: np.ndarray, kernel_frac: float = 0.05) -> np.ndarray:
    """Remove uneven lighting (phone photos) by dividing by a smooth background estimate."""
    h, w = gray.shape
    k = max(15, int(max(h, w) * kernel_frac)) | 1
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 4)
    flat = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    return np.clip(flat * 255.0, 0, 255).astype(np.uint8)


def to_ink(gray: np.ndarray) -> np.ndarray:
    """uint8 grayscale (dark ink on light paper) -> float32 ink map in [0, 1]."""
    g = gray.astype(np.float32)
    paper = np.percentile(g, 90)
    ink = np.percentile(g, 1)
    span = max(paper - ink, 32.0)
    return np.clip((paper - g) / span, 0.0, 1.0)


def trim(ink: np.ndarray, threshold: float = 0.25, margin: int = 4) -> np.ndarray:
    """Crop empty borders around the writing, keeping a small margin."""
    mask = ink > threshold
    if not mask.any():
        return ink
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, ink.shape[0])
    x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, ink.shape[1])
    return ink[y0:y1, x0:x1]


def resize_to_height(ink: np.ndarray, height: int, max_width: int, min_width: int = 16) -> np.ndarray:
    """Scale to a fixed height keeping aspect ratio; squeeze horizontally past ``max_width``."""
    h, w = ink.shape
    new_w = int(round(w * height / max(h, 1)))
    new_w = int(np.clip(new_w, min_width, max_width))
    interp = cv2.INTER_AREA if h > height else cv2.INTER_LINEAR
    out = cv2.resize(ink, (new_w, height), interpolation=interp)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def prepare_line(source, height: int, max_width: int, min_width: int = 16, flatten: bool = False) -> np.ndarray:
    """Full inference-time preprocessing for one line image."""
    gray = load_gray(source)
    if flatten:
        gray = flatten_background(gray)
    return resize_to_height(trim(to_ink(gray)), height, max_width, min_width)
