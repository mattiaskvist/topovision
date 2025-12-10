"""Engine to perform OCR on images using PaddleOCR."""

from paddleocr import PaddleOCR

from .ocr_engine import DetectionResult, OCREngine, Polygon


class PaddleOCREngine(OCREngine):
    """Engine to perform OCR on images using PaddleOCR."""

    def __init__(self):
        """Initializes PaddleOCR."""
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )

    def extract_with_polygons(self, image_path: str) -> list[DetectionResult]:
        """Extracts text and bounding polygons using PaddleOCR.

        PaddleOCR natively returns 4-point coordinates for every detection.
        """
        result = self.ocr.ocr(image_path)

        parsed_results = []

        for res in result:
            for poly, text, score in zip(
                res["rec_polys"], res["rec_texts"], res["rec_scores"], strict=True
            ):
                parsed_results.append(
                    DetectionResult(
                        text=text, polygon=Polygon(points=poly), confidence=score
                    )
                )

        return parsed_results
