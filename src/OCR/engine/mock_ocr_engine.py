"""Mock OCR engine using ground truth annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ocr_engine import DetectionResult, OCREngine, Polygon


class MockOCREngine(OCREngine):
    """Mock OCR engine that returns ground truth annotations."""

    def __init__(self, annotations_path: str | Path):
        """Initializes the mock engine.

        Args:
            annotations_path: Path to a per-image labels directory, a single
                *_labels.json file, or a COCO annotations JSON file.
        """
        self.mode = "labels_dir"
        self.labels_dir: Path | None = None
        self.labels_cache: dict[Path, list[dict[str, Any]]] = {}
        self.single_labels: list[dict[str, Any]] | None = None
        self.single_image_name: str | None = None

        self.images: dict[int, str] = {}
        self.filename_to_id: dict[str, int] = {}
        self.annotations: list[dict[str, Any]] = []
        self.anns_by_image: dict[int, list[dict[str, Any]]] = {}

        path = Path(annotations_path)
        if path.is_dir():
            self.labels_dir = path
            return

        with path.open() as f:
            data = json.load(f)

        if "labels" in data and "image" in data:
            self.mode = "labels_file"
            self.labels_dir = path.parent
            self.single_labels = data.get("labels", [])
            self.single_image_name = data.get("image", {}).get("path")
            return

        self.mode = "coco"
        self._init_coco(data)

    def _init_coco(self, data: dict[str, Any]) -> None:
        images = data.get("images", [])
        annotations = data.get("annotations", [])

        self.images = {img["id"]: img["file_name"] for img in images}
        self.filename_to_id = {v: k for k, v in self.images.items()}
        self.annotations = annotations

        for ann in annotations:
            img_id = ann.get("image_id")
            if img_id is None:
                continue
            self.anns_by_image.setdefault(img_id, []).append(ann)

    def load_labels(self, image_path: str) -> list[dict[str, Any]]:
        """Loads per-image labels if configured."""
        if self.mode not in {"labels_dir", "labels_file"}:
            return []
        return self._load_labels_for_image(image_path)

    def _load_labels_for_image(self, image_path: str) -> list[dict[str, Any]]:
        if self.mode == "labels_file":
            if (
                self.single_image_name
                and Path(image_path).name != self.single_image_name
            ):
                print(f"Warning: Image {image_path} does not match single labels file.")
                return []
            return self.single_labels or []

        if self.labels_dir is None:
            return []

        label_path = self.labels_dir / f"{Path(image_path).stem}_labels.json"
        if not label_path.exists():
            print(f"Warning: Labels file not found for {image_path}.")
            return []

        if label_path in self.labels_cache:
            return self.labels_cache[label_path]

        with label_path.open() as f:
            data = json.load(f)

        labels = data.get("labels", [])
        self.labels_cache[label_path] = labels
        return labels

    @staticmethod
    def labels_to_detections(
        labels: list[dict[str, Any]],
    ) -> list[DetectionResult]:
        """Converts label entries to mock OCR detections."""
        results = []
        for label in labels:
            elevation = label.get("elevation")
            bbox = label.get("elevation_bbox_pixels")
            if elevation is None or not bbox or len(bbox) < 4:
                continue

            x, y, w, h = bbox[:4]
            if w <= 0 or h <= 0:
                continue

            points = [
                (round(x), round(y)),
                (round(x + w), round(y)),
                (round(x + w), round(y + h)),
                (round(x), round(y + h)),
            ]

            results.append(
                DetectionResult(
                    text=str(elevation),
                    polygon=Polygon(points=points),
                    confidence=1.0,
                )
            )

        return results

    def extract_with_polygons(self, image_path: str) -> list[DetectionResult]:
        """Returns mock detection results based on the image filename."""
        if self.mode in {"labels_dir", "labels_file"}:
            labels = self._load_labels_for_image(image_path)
            return self.labels_to_detections(labels)

        return self._extract_from_coco(image_path)

    def _extract_from_coco(self, image_path: str) -> list[DetectionResult]:
        filename = Path(image_path).name
        if filename not in self.filename_to_id:
            print(f"Warning: Image {filename} not found in annotations.")
            return []

        img_id = self.filename_to_id[filename]
        anns = self.anns_by_image.get(img_id, [])

        results = []
        for ann in anns:
            text = ann.get("attributes", {}).get("text") or ann.get("text", "")
            if not text:
                continue

            points = self._polygon_from_coco(ann)
            if not points:
                continue

            results.append(
                DetectionResult(
                    text=str(text), polygon=Polygon(points=points), confidence=1.0
                )
            )

        return results

    @staticmethod
    def _polygon_from_coco(ann: dict[str, Any]) -> list[tuple[int, int]]:
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg:
            coords = seg[0]
            points = []
            for i in range(0, len(coords), 2):
                points.append((int(coords[i]), int(coords[i + 1])))
            return points

        bbox = ann.get("bbox")
        if bbox and len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            return [
                (round(x), round(y)),
                (round(x + w), round(y)),
                (round(x + w), round(y + h)),
                (round(x), round(y + h)),
            ]

        return []
