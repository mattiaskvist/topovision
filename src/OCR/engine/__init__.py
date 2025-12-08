"""Init file for the engine module."""

from .easyocr_engine import EasyOCREngine
from .ocr_engine import DetectionResult, OCREngine, Polygon
from .paddleocr_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine

__all__ = [
    "DetectionResult",
    "EasyOCREngine",
    "OCREngine",
    "PaddleOCREngine",
    "Polygon",
    "TesseractEngine",
]
