"""Engine to perform OCR on images using Tesseract."""

import pytesseract
from PIL import Image

from .ocr_engine import DetectionResult, OCREngine, Polygon


class TesseractEngine(OCREngine):
    """Engine to perform OCR on images using Tesseract."""

    def extract_with_polygons(self, image_path: str):
        """Extracts text and bounding polygons from an image using Tesseract."""
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        results = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if int(data["conf"][i]) > 0 and data["text"][i].strip():
                text = data["text"][i]
                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )

                confidence = data["conf"][i]

                # Tesseract only gives upright rectangles, but we convert
                # them to 4 points to match the interface.
                # Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
                points = [
                    (x, y),  # TL
                    (x + w, y),  # TR
                    (x + w, y + h),  # BR
                    (x, y + h),  # BL
                ]
                results.append(
                    DetectionResult(
                        text=text, polygon=Polygon(points=points), confidence=confidence
                    )
                )
        return results
