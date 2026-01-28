"""Engine to perform OCR on images using EasyOCR."""

import easyocr
import cv2

from .ocr_engine import DetectionResult, OCREngine, Polygon


class EasyOCREngine(OCREngine):
    """Engine to perform OCR on images using EasyOCR."""

    def __init__(self):
        """Initializes the EasyOCREngine with EasyOCR reader."""
        self.reader = easyocr.Reader(["en"], gpu=False)

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
        assert rotations is None, "rotations are not used now"

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
        
        # TODO Add non max suppression at the end!


        # -------

        """
        results = []
        for coord, text, conf in raw_output:
            polygon = Polygon(points=[tuple(map(int, point)) for point in coord])
            results.append(DetectionResult(text=text, polygon=polygon, confidence=conf))
        """
            
        return results
