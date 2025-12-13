"""Main pipeline for height extraction."""

import os
import random

import cv2
import numpy as np

from contour.engine.contour_engine import ContourExtractionEngine
from contour.engine.cv2_contour_engine import CV2ContourEngine
from OCR.engine.ocr_engine import OCREngine
from OCR.engine.paddleocr_engine import PaddleOCREngine

from .inference import build_adjacency_graph, infer_missing_heights
from .matcher import match_text_to_contours
from .schemas import ContourLine, HeightExtractionOutput

# Constants
CONTOUR_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # Blue
KNOWN_CONTOUR_COLOR = (0, 255, 0)  # Green
UNKNOWN_CONTOUR_COLOR = (0, 0, 255)  # Red


class HeightExtractionPipeline:
    """Pipeline to extract height curves from maps.

    Attributes:
        ocr_engine: Engine for OCR detection.
        contour_engine: Engine for contour extraction.
    """

    def __init__(
        self,
        ocr_engine: OCREngine = None,
        contour_engine: ContourExtractionEngine = None,
    ):
        """Initializes the pipeline.

        Args:
            ocr_engine: Optional custom OCR engine. Defaults to PaddleOCREngine.
            contour_engine: Optional custom contour engine. Defaults to
                CV2ContourEngine.
        """
        self.ocr_engine = ocr_engine or PaddleOCREngine()
        self.contour_engine = contour_engine or CV2ContourEngine()

    def run(
        self, image_path: str, mask_path: str, drop_ratio: float = 0.0
    ) -> HeightExtractionOutput:
        """Runs the pipeline.

        Args:
            image_path: Path to the original image.
            mask_path: Path to the binary mask of lines.
            drop_ratio: Ratio of text detections to drop (for testing inference).

        Returns:
            HeightExtractionOutput object containing extracted contours and metadata.
        """
        print(f"Processing {image_path}...")

        # 1. OCR
        print("Running OCR...")
        detections = self.ocr_engine.extract_with_polygons(image_path)
        print(f"Found {len(detections)} text detections.")

        if drop_ratio > 0:
            random.seed(42)  # Deterministic for testing
            num_keep = int(len(detections) * (1 - drop_ratio))
            detections = random.sample(detections, num_keep)
            print(f"Dropped {drop_ratio:.0%} of detections. Keeping {len(detections)}.")

        # 2. Contour Extraction
        print("Extracting contours...")
        contours = self.contour_engine.extract_contours(mask_path)
        print(f"Extracted {len(contours)} contours.")

        # 3. Matching
        print("Matching text to contours...")
        known_heights = match_text_to_contours(detections, contours)
        print(f"Matched {len(known_heights)} contours to heights.")

        # 4. Inference
        print("Inferring missing heights...")
        img = cv2.imread(image_path)
        if img is None:
            # Fallback if image read fails (shouldn't happen if OCR worked)
            # or just use mask shape
            mask = cv2.imread(mask_path)
            h, w = mask.shape[:2]
        else:
            h, w = img.shape[:2]

        adjacency = build_adjacency_graph(contours, (h, w))
        print(
            f"Adjacency graph has {sum(len(v) for v in adjacency.values()) // 2} edges."
        )
        final_heights = infer_missing_heights(contours, known_heights, adjacency)
        print(f"Inferred heights for {len(final_heights)} contours (total).")

        # Convert to Pydantic models
        contour_lines = []
        for idx, contour in enumerate(contours):
            points = [(int(p[0][0]), int(p[0][1])) for p in contour]
            height = final_heights.get(idx)

            # Determine the source of the height value explicitly:
            if idx in known_heights:
                source = "ocr"
            elif idx in final_heights:
                source = "inference"
            else:
                source = "unknown"

            contour_lines.append(
                ContourLine(id=idx, points=points, height=height, source=source)
            )

        return HeightExtractionOutput(image_path=image_path, contours=contour_lines)

    def visualize(
        self,
        output: HeightExtractionOutput,
        output_path: str,
    ):
        """Visualizes the results by drawing contours and heights on the image.

        Args:
            output: The HeightExtractionOutput object to visualize.
            output_path: Path to save the visualization.
        """
        img = cv2.imread(output.image_path)
        if img is None:
            return

        for contour_line in output.contours:
            # Convert points back to numpy array for cv2.drawContours
            pts = np.array(contour_line.points, dtype=np.int32).reshape((-1, 1, 2))

            if contour_line.height is not None:
                color = KNOWN_CONTOUR_COLOR
                label = f"{contour_line.height:.1f}"
            else:
                color = UNKNOWN_CONTOUR_COLOR
                label = "?"

            cv2.drawContours(img, [pts], -1, color, CONTOUR_THICKNESS)

            pt = contour_line.points[0]
            x = max(10, min(img.shape[1] - 50, pt[0]))
            y = max(20, min(img.shape[0] - 10, pt[1]))

            cv2.putText(
                img,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                TEXT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    # Test on synthetic data
    import glob
    from pathlib import Path

    from OCR.engine.mock_ocr_engine import MockOCREngine

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data" / "synthetic" / "perlin_noise"
    annotations_path = data_dir / "coco_annotations.json"

    # Find all image files
    image_files = sorted(glob.glob(str(data_dir / "*_image.png")))

    if not image_files:
        print("No synthetic images found.")
        exit(1)

    if annotations_path.exists():
        # Use Mock OCR and CV2 Contour Engine
        ocr_engine = MockOCREngine(str(annotations_path))
        contour_engine = CV2ContourEngine()
        pipeline = HeightExtractionPipeline(
            ocr_engine=ocr_engine, contour_engine=contour_engine
        )

        for image_file in image_files:
            image_path = Path(image_file)
            # Assuming mask has same name but _mask suffix
            mask_path = image_path.parent / image_path.name.replace(
                "_image.png", "_mask.png"
            )

            if not mask_path.exists():
                print(f"Mask not found for {image_path}, skipping.")
                continue

            print(f"\n--- Processing {image_path.name} ---")
            result = pipeline.run(str(image_path), str(mask_path), drop_ratio=0.0)

            output_path = (
                project_root
                / "output"
                / "height_extraction"
                / image_path.name.replace("_image.png", "_result.png")
            )
            pipeline.visualize(result, str(output_path))

            # Print summary stats
            total = len(result.contours)
            known = sum(1 for c in result.contours if c.source == "ocr")
            inferred = sum(1 for c in result.contours if c.source == "inference")
            print(f"Summary: {total} contours, {known} from OCR, {inferred} inferred.")

    else:
        print("Annotations not found.")
