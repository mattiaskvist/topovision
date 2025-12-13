from unittest.mock import patch

import cv2
import numpy as np
import pytest

from contour.engine.cv2_contour_engine import CV2ContourEngine


@pytest.fixture
def engine():
    return CV2ContourEngine(min_length=10.0)


def test_extract_contours_file_not_found(engine):
    with patch("cv2.imread", return_value=None), pytest.raises(FileNotFoundError):
        engine.extract_contours("nonexistent.png")


def test_extract_contours_empty_mask(engine):
    # Create a black image (no contours)
    mask = np.zeros((100, 100), dtype=np.uint8)

    with patch("cv2.imread", return_value=mask):
        contours = engine.extract_contours("empty.png")
        assert len(contours) == 0


def test_extract_contours_valid_mask(engine):
    # Create an image with a simple square
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

    with patch("cv2.imread", return_value=mask):
        contours = engine.extract_contours("square.png")
        assert len(contours) == 1
        # Check if contour is roughly square (4 points approx)
        # approxPolyDP might reduce points, but for a perfect square should be small
        assert len(contours[0]) >= 4


def test_extract_contours_min_length_filter(engine):
    # Create two contours: one large, one small
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (180, 180), 255, -1)  # Large
    cv2.rectangle(mask, (10, 10), (12, 12), 255, -1)  # Tiny

    # Set min_length to filter out the tiny one
    engine.min_length = 50.0

    with patch("cv2.imread", return_value=mask):
        contours = engine.extract_contours("mixed.png")
        # Should only keep the large one
        assert len(contours) == 1
