"""Engine to perform OCR on images."""

from abc import ABC, abstractmethod


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_with_polygons(self, image_path: str):
        """Returns: List[ (text, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]) ].

        We return 4 corners to support rotated/diagonal text boxes.
        """
        pass
