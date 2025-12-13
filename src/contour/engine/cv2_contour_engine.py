"""CV2 implementation of contour extraction engine."""

import cv2
import numpy as np

from .contour_engine import ContourExtractionEngine


class CV2ContourEngine(ContourExtractionEngine):
    """Extracts contours using OpenCV.

    Attributes:
        min_length: Minimum length of a contour to be kept.
    """

    def __init__(self, min_length: float = 50.0):
        """Initializes the CV2 contour engine.

        Args:
            min_length: Minimum length of a contour to be kept.
        """
        self.min_length = min_length

    def extract_contours(self, mask_path: str) -> list[np.ndarray]:
        """Extracts contours from a binary mask file.

        Args:
            mask_path: Path to the binary mask image.

        Returns:
            List of contours, where each contour is a numpy array of shape (N, 1, 2).

        Raises:
            FileNotFoundError: If the mask file cannot be read.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Morphological closing to merge close lines
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # RETR_LIST: Retrieve all contours without establishing hierarchy
        # CHAIN_APPROX_SIMPLE: Compress horizontal, vertical, and diagonal segments
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)

            if length >= self.min_length:
                epsilon = 0.005 * length
                approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
                filtered_contours.append(approx)

        return filtered_contours
