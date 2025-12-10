"""Module for extracting contours from binary masks."""

import cv2
import numpy as np


def extract_contours(mask_path: str, min_length: float = 50.0) -> list[np.ndarray]:
    """Extracts contours from a binary mask file.

    Args:
        mask_path: Path to the binary mask image.
        min_length: Minimum length of a contour to be kept.

    Returns:
        List of contours, where each contour is a numpy array of shape (N, 1, 2).
    """
    # Read image as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at {mask_path}")

    # Ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    # RETR_LIST: Retrieve all contours without establishing hierarchy
    # CHAIN_APPROX_SIMPLE: Compress horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for cnt in contours:
        # Calculate arc length
        length = cv2.arcLength(cnt, closed=False)  # Open contours for height lines?
        # Height lines are usually curves, but might be closed loops (hills).
        # Let's assume they can be either, but we treat them as polylines.

        if length >= min_length:
            # Simplify contour
            epsilon = 0.005 * length
            approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
            filtered_contours.append(approx)

    return filtered_contours
