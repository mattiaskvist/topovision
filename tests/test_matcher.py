import math

import numpy as np
import pytest

from height_extraction.matcher import (
    calculate_angle,
    calculate_centroid,
    match_text_to_contours,
    min_distance_to_contour,
    parse_height_text,
    point_line_segment_distance,
)
from OCR.engine.ocr_engine import DetectionResult, Polygon

# --- Helper Function Tests ---


def test_calculate_centroid():
    points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    centroid = calculate_centroid(points)
    assert centroid == (5.0, 5.0)


def test_calculate_centroid_empty():
    with pytest.raises(ValueError):
        calculate_centroid([])


def test_point_line_segment_distance():
    # Point on line
    assert point_line_segment_distance(5, 0, 0, 0, 10, 0) == 0.0
    # Point off line (perpendicular)
    assert point_line_segment_distance(5, 5, 0, 0, 10, 0) == 5.0
    # Point beyond endpoint
    assert point_line_segment_distance(15, 0, 0, 0, 10, 0) == 5.0
    # Zero length segment
    assert point_line_segment_distance(5, 5, 2, 2, 2, 2) == math.hypot(3, 3)


def test_calculate_angle():
    # Horizontal right
    assert calculate_angle((0, 0), (10, 0)) == 0.0
    # Vertical up
    assert calculate_angle((0, 0), (0, 10)) == 90.0
    # Horizontal left
    assert calculate_angle((10, 0), (0, 0)) == 180.0
    # Vertical down
    assert calculate_angle((0, 10), (0, 0)) == -90.0


def test_parse_height_text():
    assert parse_height_text("100") == 100.0
    assert parse_height_text("100m") == 100.0
    assert parse_height_text("100.5") == 100.5
    assert parse_height_text("-50") == -50.0
    assert parse_height_text("invalid") is None
    assert parse_height_text("") is None


def test_min_distance_to_contour():
    # Contour: horizontal line from (0,0) to (10,0)
    contour = np.array([[[0, 0]], [[10, 0]]], dtype=np.int32)

    # Point at (5, 5) -> dist 5, angle 0
    dist, angle = min_distance_to_contour((5, 5), contour)
    assert dist == 5.0
    assert angle == 0.0

    # Point at (15, 0) -> dist 5, angle 0 (closest segment is still 0-10)
    dist, angle = min_distance_to_contour((15, 0), contour)
    assert dist == 5.0
    assert angle == 0.0


# --- Matcher Logic Tests ---


def create_detection(text, points):
    return DetectionResult(text=text, polygon=Polygon(points=points), confidence=1.0)


def test_match_text_to_contours_basic():
    # Contour: vertical line at x=10
    contour = np.array([[[10, 0]], [[10, 100]]], dtype=np.int32)
    contours = [contour]

    # Text "100" at (15, 50), vertical orientation
    # Points: top-left, top-right, bottom-right, bottom-left
    # Vertical text: p0=(15, 40), p1=(15, 60) -> angle 90 (down to up? no p0 is start)
    # Let's say text is written bottom-to-top or top-to-bottom?
    # calculate_angle uses p0->p1.
    # If text is vertical, p0 and p1 should form vertical line.
    points = [(15, 40), (15, 60), (20, 60), (20, 40)]
    # Centroid: (17.5, 50). Dist to line x=10 is 7.5.
    # Angle p0->p1: (0, 20) -> 90 degrees.
    # Contour angle: (0, 100) -> 90 degrees.

    detection = create_detection("100", points)

    matches = match_text_to_contours([detection], contours)

    assert 0 in matches
    assert matches[0] == 100.0


def test_match_text_to_contours_distance_mismatch():
    contour = np.array([[[10, 0]], [[10, 100]]], dtype=np.int32)
    contours = [contour]

    # Text far away at x=100
    points = [(100, 40), (100, 60), (105, 60), (105, 40)]
    detection = create_detection("100", points)

    matches = match_text_to_contours([detection], contours, max_distance=50.0)

    assert 0 not in matches


def test_match_text_to_contours_angle_mismatch():
    # Contour: vertical line (angle 90)
    contour = np.array([[[10, 0]], [[10, 100]]], dtype=np.int32)
    contours = [contour]

    # Text horizontal (angle 0)
    points = [(15, 50), (25, 50), (25, 55), (15, 55)]
    detection = create_detection("100", points)

    # Angle diff: 90. Threshold 30. Should fail.
    matches = match_text_to_contours([detection], contours, max_angle_diff=30.0)

    assert 0 not in matches


def test_match_text_to_contours_multiple_contours():
    # c1: x=10
    c1 = np.array([[[10, 0]], [[10, 100]]], dtype=np.int32)
    # c2: x=20
    c2 = np.array([[[20, 0]], [[20, 100]]], dtype=np.int32)
    contours = [c1, c2]

    # Text at x=18 (closer to c2)
    points = [(18, 40), (18, 60), (22, 60), (22, 40)]
    detection = create_detection("200", points)

    matches = match_text_to_contours([detection], contours)

    assert 0 not in matches
    assert 1 in matches
    assert matches[1] == 200.0


def test_match_text_to_contours_empty():
    assert match_text_to_contours([], []) == {}

    contour = np.array([[[10, 0]], [[10, 100]]], dtype=np.int32)
    assert match_text_to_contours([], [contour]) == {}

    points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    detection = create_detection("100", points)
    assert match_text_to_contours([detection], []) == {}
