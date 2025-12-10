"""Mock OCR engine using COCO annotations."""

import json
import os

from .ocr_engine import DetectionResult, OCREngine, Polygon


class MockOCREngine(OCREngine):
    """Mock OCR engine that returns ground truth annotations."""

    def __init__(self, annotations_path: str):
        """Initializes the mock engine.

        Args:
            annotations_path: Path to the COCO annotations JSON file.
        """
        with open(annotations_path) as f:
            data = json.load(f)

        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        self.filename_to_id = {v: k for k, v in self.images.items()}
        self.annotations = data["annotations"]

        # Pre-group annotations by image_id
        self.anns_by_image = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.anns_by_image:
                self.anns_by_image[img_id] = []
            self.anns_by_image[img_id].append(ann)

    def extract_with_polygons(self, image_path: str) -> list[DetectionResult]:
        """Returns mock detection results based on the image filename."""
        filename = os.path.basename(image_path)

        if filename not in self.filename_to_id:
            print(f"Warning: Image {filename} not found in annotations.")
            return []

        img_id = self.filename_to_id[filename]
        anns = self.anns_by_image.get(img_id, [])

        results = []
        for ann in anns:
            text = ann.get("attributes", {}).get("text", "")
            if not text:
                continue

            # Segmentation is [[x1, y1, x2, y2, ...]]
            seg = ann.get("segmentation", [[]])[0]
            points = []
            for i in range(0, len(seg), 2):
                points.append((int(seg[i]), int(seg[i + 1])))

            results.append(
                DetectionResult(
                    text=text, polygon=Polygon(points=points), confidence=1.0
                )
            )

        return results
