"""Engine to perform OCR on images using EasyOCR."""

import easyocr

from .ocr_engine import OCREngine


class EasyOCREngine(OCREngine):
    """Engine to perform OCR on images using EasyOCR."""

    def __init__(self):
        """Initializes the EasyOCREngine with EasyOCR reader."""
        self.reader = easyocr.Reader(["en"], gpu=False)

    def extract_with_polygons(self, image_path: str, rotations=None):
        """Extract text with polygons.

        Args:
            image_path (str): Path to image.
            rotations (list): List of angles to check. Defaults to [90, 180, 270].
                              Note: Adding angles increases processing time.
        """
        if rotations is None:  # Default rotations if none provided
            rotations = [90, 180, 270]

        raw_output = self.reader.readtext(image_path, rotation_info=rotations)

        results = []
        for coord, text, _conf in raw_output:
            polygon = [tuple(map(int, point)) for point in coord]
            results.append((text, polygon))

        return results
