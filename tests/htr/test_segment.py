import numpy as np
import pytest

from legible.htr.segment import find_lines, segment_lines

from .helpers import synthetic_page

TEXTS = ["The first line of my notes", "then a second line here", "and a third one at the end", "four"]


@pytest.mark.parametrize("drift", [0.0, 0.04, -0.04])
def test_finds_each_line_in_order_even_when_lines_drift(drift):
    page = synthetic_page(TEXTS, drift=drift, seed=2)
    lines = segment_lines(page)
    assert len(lines) == len(TEXTS)
    tops = [line.bbox[1] for line in lines]
    assert tops == sorted(tops)
    for line in lines:
        x0, y0, x1, y1 = line.bbox
        assert 0 <= x0 < x1 <= page.shape[1] and 0 <= y0 < y1 <= page.shape[0]
        assert line.ink.max() > 0.5


def test_ruled_paper_lines_are_not_text():
    page = synthetic_page(TEXTS[:2], seed=4).astype(np.int16)
    for y in range(20, page.shape[0], 37):
        page[y : y + 2, :] -= 60  # faint blue-ish rules across the whole sheet
    lines = segment_lines(np.clip(page, 0, 255).astype(np.uint8))
    assert len(lines) == 2


def test_large_photos_map_boxes_back_to_source_pixels():
    import cv2

    page = synthetic_page(TEXTS[:2], seed=1)
    big = cv2.resize(page, (page.shape[1] * 3, page.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    small_boxes = [l.bbox for l in segment_lines(page)]
    big_boxes = [l.bbox for l in segment_lines(big, max_side=page.shape[1])]
    assert len(small_boxes) == len(big_boxes) == 2
    for (a0, b0, a1, b1), (c0, d0, c1, d1) in zip(small_boxes, big_boxes):
        assert abs(c0 - 3 * a0) <= 12 and abs(d1 - 3 * b1) <= 12


def test_modes_and_blank_images():
    page = synthetic_page(TEXTS[:1], seed=3)
    assert len(find_lines(page, "auto")) == 1
    assert len(find_lines(synthetic_page(TEXTS, seed=3), "line")) == 1
    blank = np.full((200, 300), 230, np.uint8)
    assert find_lines(blank, "page") == [] and find_lines(blank, "auto") == []
