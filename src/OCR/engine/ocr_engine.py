"""Engine to perform OCR on images."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class Polygon(BaseModel):
    """Polygon coordinates."""

    points: list[tuple[int, int]]


class DetectionResult(BaseModel):
    """Result of a text detection."""

    text: str
    polygon: Polygon
    confidence: float


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_with_polygons(self, image_path: str) -> list[DetectionResult]:
        """Returns: List[DetectionResult].

        We return 4 corners to support rotated/diagonal text boxes.
        """
        pass
