"""Abstract base class for contour extraction engines."""

from abc import ABC, abstractmethod

import numpy as np


class ContourExtractionEngine(ABC):
    """Abstract base class for contour extraction engines.

    This class defines the interface for extracting contours from binary masks.
    """

    @abstractmethod
    def extract_contours(self, mask_path: str) -> list[np.ndarray]:
        """Extracts contours from a binary mask file.

        Args:
            mask_path: Path to the binary mask image.

        Returns:
            List of contours, where each contour is a numpy array of shape (N, 1, 2).
        """
        pass
