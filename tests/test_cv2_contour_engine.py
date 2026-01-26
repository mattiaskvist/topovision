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
    # Create an image with a thick line
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a line from (10, 50) to (90, 50) with thickness 5
    cv2.line(mask, (10, 50), (90, 50), 255, 5)

    with patch("cv2.imread", return_value=mask):
        contours = engine.extract_contours("line.png")
        assert len(contours) == 1
        # Check if contour is roughly linear
        # For a line of length 80, arcLength should be around 80 (or 160 if loop)
        # With skeletonization, it should be around 80.
        length = cv2.arcLength(contours[0], False)
        assert length > 70


def test_extract_contours_min_length_filter(engine):
    # Create two lines: one large, one small
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (20, 20), (180, 20), 255, 5)  # Large (len 160)
    cv2.line(mask, (10, 50), (12, 50), 255, 5)  # Tiny (len 2)

    # Set min_length to filter out the tiny one
    engine.min_length = 50.0

    with patch("cv2.imread", return_value=mask):
        contours = engine.extract_contours("mixed.png")
        # Should only keep the large one
        assert len(contours) == 1


def test_skeletonize_thick_line(engine):
    # Create a thick line (filled rectangle)
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a thick line from (10, 50) to (90, 50) with thickness 10
    cv2.line(mask, (10, 50), (90, 50), 255, 10)

    # Without skeletonization, this would be a rectangle with area ~ 80*10 = 800
    # With skeletonization, it should be a thin line.

    # We can test the private method directly or via extract_contours
    # Let's test the private method logic first to ensure it thins
    skeleton = engine.skeletonize(mask)

    # Count non-zero pixels.
    # Original: ~80 * 10 = 800 pixels
    # Skeleton: ~80 pixels (length of line)

    original_pixels = cv2.countNonZero(mask)
    skeleton_pixels = cv2.countNonZero(skeleton)

    assert original_pixels > 700
    assert skeleton_pixels < 100  # Should be much less
    assert skeleton_pixels > 50  # But not empty
