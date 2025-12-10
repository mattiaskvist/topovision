"""Main pipeline for height extraction."""

import os

import cv2
import numpy as np

from OCR.engine.ocr_engine import OCREngine
from OCR.engine.paddleocr_engine import PaddleOCREngine

from .contours import extract_contours
from .inference import build_adjacency_graph, infer_missing_heights
from .matcher import match_text_to_contours


class HeightExtractionPipeline:
    """Pipeline to extract height curves from maps."""

    def __init__(self, ocr_engine: OCREngine = None):
        """Initializes the pipeline."""
        self.ocr_engine = ocr_engine or PaddleOCREngine()

    def run(
        self, image_path: str, mask_path: str, drop_ratio: float = 0.0
    ) -> tuple[list[np.ndarray], dict[int, float]]:
        """Runs the pipeline.

        Args:
            image_path: Path to the original image.
            mask_path: Path to the binary mask of lines.
            drop_ratio: Ratio of text detections to drop (for testing inference).

        Returns:
            Tuple of (contours, height_map) where height_map maps contour idx to height.
        """
        print(f"Processing {image_path}...")

        # 1. OCR
        print("Running OCR...")
        detections = self.ocr_engine.extract_with_polygons(image_path)
        print(f"Found {len(detections)} text detections.")

        if drop_ratio > 0:
            import random

            random.seed(42)  # Deterministic for testing
            num_keep = int(len(detections) * (1 - drop_ratio))
            detections = random.sample(detections, num_keep)
            print(f"Dropped {drop_ratio:.0%} of detections. Keeping {len(detections)}.")

        # 2. Contour Extraction
        print("Extracting contours...")
        contours = extract_contours(mask_path)
        print(f"Extracted {len(contours)} contours.")

        # 3. Matching
        print("Matching text to contours...")
        known_heights = match_text_to_contours(detections, contours)
        print(f"Matched {len(known_heights)} contours to heights.")

        # 4. Inference
        print("Inferring missing heights...")
        # Need image shape for adjacency
        img = cv2.imread(image_path)
        if img is None:
            # Fallback if image read fails (shouldn't happen if OCR worked)
            # or just use mask shape
            mask = cv2.imread(mask_path)
            h, w = mask.shape[:2]
        else:
            h, w = img.shape[:2]

        adjacency = build_adjacency_graph(contours, (h, w), max_dist=50.0)
        print(
            f"Adjacency graph has {sum(len(v) for v in adjacency.values()) // 2} edges."
        )
        final_heights = infer_missing_heights(contours, known_heights, adjacency)
        print(f"Inferred heights for {len(final_heights)} contours (total).")

        return contours, final_heights

    def visualize(
        self,
        image_path: str,
        contours: list[np.ndarray],
        height_map: dict[int, float],
        output_path: str,
    ):
        """Visualizes the results."""
        img = cv2.imread(image_path)
        if img is None:
            return

        # Draw contours
        for idx, contour in enumerate(contours):
            height = height_map.get(idx)

            if height is not None:
                # Color based on height (simple cycle)
                color = (0, 255, 0)  # Green for known/inferred
                label = f"{height:.1f}"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "?"

            cv2.drawContours(img, [contour], -1, color, 2)

            # Draw label at the first point of the contour
            pt = contour[0][0]
            x = max(10, min(img.shape[1] - 50, pt[0]))
            y = max(20, min(img.shape[0] - 10, pt[1]))

            cv2.putText(
                img,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),  # Blue text
                1,
                cv2.LINE_AA,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    # Test on synthetic data
    from pathlib import Path

    from OCR.engine.mock_ocr_engine import MockOCREngine

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data" / "synthetic" / "perlin_noise"
    annotations_path = data_dir / "coco_annotations.json"

    image_path = data_dir / "sparse_0_image.png"
    mask_path = data_dir / "sparse_0_mask.png"

    if image_path.exists() and mask_path.exists() and annotations_path.exists():
        # Use Mock OCR
        ocr_engine = MockOCREngine(str(annotations_path))
        pipeline = HeightExtractionPipeline(ocr_engine=ocr_engine)

        contours, heights = pipeline.run(
            str(image_path), str(mask_path), drop_ratio=0.2
        )

        output_path = (
            project_root / "output" / "height_extraction" / "sparse_0_result.png"
        )
        pipeline.visualize(str(image_path), contours, heights, str(output_path))
    else:
        print("Data not found.")
