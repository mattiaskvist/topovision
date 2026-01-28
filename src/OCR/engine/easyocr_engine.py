"""Engine to perform OCR on images using EasyOCR."""

import easyocr
import cv2
import numpy as np

from .ocr_engine import DetectionResult, OCREngine, Polygon


class EasyOCREngine(OCREngine):
    """Engine to perform OCR on images using EasyOCR."""

    def __init__(self):
        """Initializes the EasyOCREngine with EasyOCR reader."""
        self.reader = easyocr.Reader(["en"], gpu=False)

    def _nms(self, results: list[DetectionResult], iou_thresh: float = 0.3) -> list[DetectionResult]:
        """Non-maximum suppression to remove overlapping detections."""
        if not results:
            return []
        
        # Sort by confidence descending
        results = sorted(results, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        for result in results:
            # Get bounding box from polygon points
            points = result.polygon.points
            x1 = min(p[0] for p in points)
            y1 = min(p[1] for p in points)
            x2 = max(p[0] for p in points)
            y2 = max(p[1] for p in points)
            
            overlap = False
            for kept in keep:
                # Get bounding box from kept polygon
                kept_points = kept.polygon.points
                kx1 = min(p[0] for p in kept_points)
                ky1 = min(p[1] for p in kept_points)
                kx2 = max(p[0] for p in kept_points)
                ky2 = max(p[1] for p in kept_points)
                
                # Calculate IoU
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                
                if iou > iou_thresh:
                    overlap = True
                    break
            
            if not overlap:
                keep.append(result)
        
        return keep

    def _preprocess(self, img: np.ndarray) -> list[np.ndarray]:
        """Generate multiple preprocessed versions for better detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Original grayscale
        variants = [gray]
        
        # Enhanced contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))
        
        # Thresholded binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(binary)
        
        return variants

    def extract_with_polygons(
        self, image_path: str, rotations=None, scale_factor: float = 2.5
    ) -> list[DetectionResult]:
        """Extract text with polygons.

        Args:
            image_path (str): Path to image.
            rotations (list): List of angles to check. Defaults to [90, 180, 270].
                              Note: Adding angles increases processing time.
        """
        if rotations is not None:
            print("Warning, rotations are not used now!!")

        img_orig = cv2.imread(str(image_path))
        if img_orig is None:
            return []
            
        # Resize upwards to improve OCR on small text
        h, w = img_orig.shape[:2]
        img = cv2.resize(img_orig, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LANCZOS4)
        
        variants = self._preprocess(img)
        
        results = []
        for variant in variants:
            outputs = self.reader.readtext(
                variant,
                allowlist='0123456789',
                paragraph=False,
                min_size=8,
                text_threshold=0.4,
                low_text=0.3,
            )
            
            for bbox, text, conf in outputs:
                if not text or conf < 0.35:
                    continue
                    
                # Scale coordinates back to original image size
                scaled_points = [
                    (int(p[0] / scale_factor), int(p[1] / scale_factor)) 
                    for p in bbox
                ]
                polygon = Polygon(points=scaled_points)
                results.append(DetectionResult(text=text, polygon=polygon, confidence=conf))
        
        results = self._nms(results)
        return results
