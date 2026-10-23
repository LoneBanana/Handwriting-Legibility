"""Finds the text lines in a photo of handwriting.

The classic approach (one horizontal projection profile over the whole page) assumes
straight, evenly spaced lines. Dysgraphic handwriting often drifts uphill or downhill,
waves, and varies in size, so here lines are *tracked* across the page instead:

1. normalise lighting, binarise, and erase ruled notebook lines and margin lines;
2. blur the ink strongly sideways so the words of a line melt into one ridge;
3. in narrow vertical strips, find the ridge peaks (line centres);
4. link peaks from strip to strip into line paths that may climb, fall, or curve;
5. give every connected ink component (letters, words, i-dots) to the nearest path;
6. cut each line out with a mask, so ascenders/descenders of neighbouring lines that
   reach into its bounding box are not read as part of it.

Everything is classical image processing (OpenCV/SciPy), fast on a CPU, and needs no
training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import find_peaks

from .data.image import flatten_background, to_ink, trim


@dataclass
class LineCrop:
    ink: np.ndarray                          # float32 ink map (ink=1) of this line only
    bbox: tuple[int, int, int, int]          # x0, y0, x1, y1 in source-image pixels


@dataclass
class _Path:
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    misses: int = 0

    def y_at(self, x: float) -> float:
        return float(np.interp(x, self.xs, self.ys))


def prepare_page(gray: np.ndarray, max_side: int = 2000) -> tuple[np.ndarray, float]:
    """Downscales very large photos and returns (ink map, scale applied)."""
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return to_ink(flatten_background(gray)), scale


def _binarize(ink: np.ndarray) -> np.ndarray:
    u8 = (np.clip(ink, 0, 1) * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return u8 > max(threshold, 64)


def _remove_rules(binary: np.ndarray) -> np.ndarray:
    """Erases long straight horizontal/vertical lines (ruled paper, margins, table borders)."""
    h, w = binary.shape
    u8 = binary.astype(np.uint8)
    horizontal = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1)))
    vertical = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 4))))
    rules = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    return binary & (rules == 0)


def _text_height(stats: np.ndarray) -> float:
    """Typical letter height: median height of components that look like letters or words."""
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    candidates = heights[(areas >= 12) & (heights >= 4)]
    if candidates.size == 0:
        return 0.0
    return float(np.median(candidates))


def _track_lines(blurred: np.ndarray, text_h: float) -> list[_Path]:
    h, w = blurred.shape
    strip = int(max(4 * text_h, w / 24, 16))
    peak_gap = max(int(0.7 * text_h), 3)
    tolerance = 0.8 * text_h
    active: list[_Path] = []
    finished: list[_Path] = []
    for x0 in range(0, w, strip):
        x1 = min(x0 + strip, w)
        profile = blurred[:, x0:x1].mean(axis=1)
        top = float(profile.max())
        if top <= 1e-4:
            peaks = np.array([], dtype=int)
        else:
            peaks, _ = find_peaks(profile, distance=peak_gap, prominence=0.25 * top, height=0.12 * top)
        cx = (x0 + x1) / 2
        used: set[int] = set()
        # Greedily continue existing lines with the nearest unused peak.
        pairs = sorted(
            (abs(path.y_at(cx) - float(p)), i, j)
            for i, path in enumerate(active)
            for j, p in enumerate(peaks)
        )
        matched_paths: set[int] = set()
        for distance, i, j in pairs:
            if distance > tolerance or i in matched_paths or j in used:
                continue
            active[i].xs.append(cx)
            active[i].ys.append(float(peaks[j]))
            active[i].misses = 0
            matched_paths.add(i)
            used.add(j)
        still_active = []
        for i, path in enumerate(active):
            if i not in matched_paths:
                path.misses += 1
            (still_active if path.misses <= 2 else finished).append(path)
        active = still_active
        for j, p in enumerate(peaks):
            if j not in used:
                active.append(_Path([cx], [float(p)]))
    return finished + active


def segment_lines(gray: np.ndarray, max_side: int = 2000) -> list[LineCrop]:
    """Splits a page photo (uint8 grayscale) into line crops, top to bottom."""
    ink, scale = prepare_page(gray, max_side)
    binary = _remove_rules(_binarize(ink))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return []
    text_h = _text_height(stats)
    if text_h <= 0:
        return []

    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=2.5 * text_h, sigmaY=0.3 * text_h)
    paths = [p for p in _track_lines(blurred, text_h) if len(p.xs) >= 1]
    if not paths:
        return []

    # Assign each ink component to the path nearest its centre, at the component's x.
    members: dict[int, list[int]] = {i: [] for i in range(len(paths))}
    for c in range(1, count):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        cx, cy = centroids[c]
        distances = [abs(path.y_at(cx) - cy) for path in paths]
        best = int(np.argmin(distances))
        if distances[best] > 2.5 * text_h and area < text_h * text_h:
            continue  # a speck far from any line
        members[best].append(c)

    lines: list[tuple[float, LineCrop]] = []
    margin = max(2, int(0.2 * text_h))
    kernel = np.ones((3, 3), np.uint8)
    for i, comps in members.items():
        if not comps:
            continue
        ink_area = float(stats[comps, cv2.CC_STAT_AREA].sum())
        if ink_area < 0.5 * text_h * text_h:
            continue
        x0 = int(stats[comps, cv2.CC_STAT_LEFT].min())
        y0 = int(stats[comps, cv2.CC_STAT_TOP].min())
        x1 = int((stats[comps, cv2.CC_STAT_LEFT] + stats[comps, cv2.CC_STAT_WIDTH]).max())
        y1 = int((stats[comps, cv2.CC_STAT_TOP] + stats[comps, cv2.CC_STAT_HEIGHT]).max())
        x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
        x1, y1 = min(x1 + margin, ink.shape[1]), min(y1 + margin, ink.shape[0])
        window = labels[y0:y1, x0:x1]
        mask = np.isin(window, comps).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)  # keep the soft anti-aliased edges of strokes
        crop = ink[y0:y1, x0:x1] * mask
        bbox = (int(x0 / scale), int(y0 / scale), int(np.ceil(x1 / scale)), int(np.ceil(y1 / scale)))
        order = float(np.median(paths[i].ys))
        lines.append((order, LineCrop(crop.astype(np.float32), bbox)))
    lines.sort(key=lambda item: item[0])
    return [crop for _, crop in lines]


def single_line(gray: np.ndarray, max_side: int = 4000) -> list[LineCrop]:
    """Treats the whole image as one line (``mode="line"``)."""
    ink, scale = prepare_page(gray, max_side)
    mask = ink > 0.25
    if not mask.any():
        return []
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    bbox = (int(x0 / scale), int(y0 / scale), int(np.ceil(x1 / scale)), int(np.ceil(y1 / scale)))
    return [LineCrop(trim(ink), bbox)]


def find_lines(gray: np.ndarray, mode: str = "auto") -> list[LineCrop]:
    """``line``: the whole image is one line; ``page``: always segment; ``auto``: segment,
    but treat short wide images with at most one detected line as a single line."""
    if mode == "line":
        return single_line(gray)
    lines = segment_lines(gray)
    if mode == "auto" and len(lines) <= 1:
        h, w = gray.shape
        if not lines or w >= 2 * h:
            return single_line(gray)
    return lines
