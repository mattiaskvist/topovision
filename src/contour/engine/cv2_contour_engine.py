"""CV2 implementation of contour extraction engine."""

import cv2
import numpy as np

from .contour_engine import ContourExtractionEngine


class CV2ContourEngine(ContourExtractionEngine):
    """Extracts contours using OpenCV.

    Attributes:
        min_length: Minimum length of a contour to be kept.
        threshold_value: Threshold value for binarization.
        threshold_max_value: Max value for binarization.
        morph_kernel_size: Kernel size for morphological operations.
        morph_iterations: Number of iterations for morphological operations.
        epsilon_factor: Factor for approximation accuracy.
    """

    def __init__(
        self,
        min_length: float = 50.0,
        threshold_value: int = 127,
        threshold_max_value: int = 255,
        morph_kernel_size: tuple[int, int] = (3, 3),
        morph_iterations: int = 2,
        epsilon_factor: float = 0.005,
    ):
        """Initializes the CV2 contour engine.

        Args:
            min_length: Minimum length of a contour to be kept.
            threshold_value: Threshold value for binarization.
            threshold_max_value: Max value for binarization.
            morph_kernel_size: Kernel size for morphological operations.
            morph_iterations: Number of iterations for morphological operations.
            epsilon_factor: Factor for approximation accuracy.
        """
        self.min_length = min_length
        self.threshold_value = threshold_value
        self.threshold_max_value = threshold_max_value
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.epsilon_factor = epsilon_factor

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

        _, binary = cv2.threshold(
            mask, self.threshold_value, self.threshold_max_value, cv2.THRESH_BINARY
        )

        # Morphological closing to merge close lines
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations
        )

        # RETR_LIST: Retrieve all contours without establishing hierarchy
        # CHAIN_APPROX_SIMPLE: Compress horizontal, vertical, and diagonal segments
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)

            if length >= self.min_length:
                epsilon = self.epsilon_factor * length
                approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
                filtered_contours.append(approx)

        return filtered_contours
