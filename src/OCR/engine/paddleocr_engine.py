"""Engine to perform OCR on images using PaddleOCR."""

from __future__ import annotations

from .ocr_engine import DetectionResult, OCREngine, Polygon


class PaddleOCREngine(OCREngine):
    """Engine to perform OCR on images using PaddleOCR."""

    def __init__(self):
        """Initializes PaddleOCR."""
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang="en",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )

    @staticmethod
    def _unwrap_res(obj):
        """Unwrap PaddleOCR 3.x result objects to a plain dict when possible."""
        # OCRResult-like: has attribute .res
        if hasattr(obj, "res") and isinstance(obj.res, dict):
            obj = obj.res

        # Some outputs: {"res": {...}}
        if isinstance(obj, dict) and "res" in obj and isinstance(obj["res"], dict):
            obj = obj["res"]

        return obj

    def extract_with_polygons(self, image_path: str) -> list[DetectionResult]:
        """Extracts text and bounding polygons using PaddleOCR."""
        if hasattr(self.ocr, "predict"):
            raw = self.ocr.predict(input=image_path)
        else:
            raw = self.ocr.ocr(image_path)

        parsed_results: list[DetectionResult] = []

        for item in raw:
            res = self._unwrap_res(item)
            if not isinstance(res, dict):
                continue

            texts = res.get("rec_texts", []) or []
            scores = res.get("rec_scores", []) or []

            polys = (
                res.get("rec_polys")
                or res.get("textline_polys")
                or res.get("dt_polys")
                or res.get("det_polys")
                or []
            )

            n = min(len(polys), len(texts), len(scores))
            for i in range(n):
                poly = polys[i]
                text = texts[i]
                score = scores[i]

                poly_pts = poly.tolist() if hasattr(poly, "tolist") else poly

                parsed_results.append(
                    DetectionResult(
                        text=str(text),
                        polygon=Polygon(points=poly_pts),
                        confidence=float(score),
                    )
                )

        return parsed_results
