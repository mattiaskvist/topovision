"""Verify contour matching using ground-truth labels."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Make "src/" importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from height_extraction.matcher import match_text_to_contours  # noqa: E402
from OCR.engine.mock_ocr_engine import MockOCREngine  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data" / "training" / "N60E013" / "N60E013"


def _image_path_from_label(label_path: Path) -> Path:
    stem = label_path.stem
    if stem.endswith("_labels"):
        stem = stem[: -len("_labels")]
    return label_path.with_name(f"{stem}.png")


def _labels_to_contours(
    labels: list[dict],
) -> tuple[list[dict], list[np.ndarray]]:
    filtered: list[dict] = []
    contours: list[np.ndarray] = []

    for label in labels:
        bbox = label.get("elevation_bbox_pixels")
        contour_pixels = label.get("contour_line_pixels")
        elevation = label.get("elevation")
        if (
            elevation is None
            or not bbox
            or not contour_pixels
            or len(contour_pixels) < 2
        ):
            continue

        points = [(round(p[0]), round(p[1])) for p in contour_pixels if len(p) >= 2]
        if len(points) < 2:
            continue

        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        filtered.append(label)
        contours.append(contour)

    return filtered, contours


def _save_overlay(
    image_path: Path,
    labels: list[dict],
    contours: list[np.ndarray],
    matches: dict[int, float],
    output_path: Path,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        return

    for idx, contour in enumerate(contours):
        expected = float(labels[idx]["elevation"])
        predicted = matches.get(idx)
        correct = predicted is not None and math.isclose(
            float(predicted), expected, abs_tol=0.01
        )

        if predicted is None:
            label_text = f"{expected:.0f} ?"
        elif correct:
            label_text = f"{expected:.0f}"
        else:
            label_text = f"{expected:.0f}->{predicted:.0f}"

        color = (0, 255, 0) if correct else (0, 0, 255)
        cv2.polylines(img, [contour], False, color, 2)

        bbox = labels[idx].get("elevation_bbox_pixels", [])
        if len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            x1, y1 = round(x), round(y)
            x2, y2 = round(x + w), round(y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        pt = contour[0][0]
        cv2.putText(
            img,
            label_text,
            (int(pt[0]), int(pt[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def main() -> int:
    """Run main program."""
    parser = argparse.ArgumentParser(
        description="Verify contour matching using per-image label annotations."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Directory containing images and *_labels.json files.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max images to process.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Max distance for matching.",
    )
    parser.add_argument(
        "--max-angle-diff",
        type=float,
        default=30.0,
        help="Max angle difference for matching.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save mismatch overlays.",
    )
    parser.add_argument(
        "--max-overlays",
        type=int,
        default=5,
        help="Max overlays to save.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 2

    label_files = sorted(data_dir.glob("*_labels.json"))
    if not label_files:
        print(f"No label files found in {data_dir}")
        return 2

    engine = MockOCREngine(data_dir)

    total_labels = 0
    total_matched = 0
    total_correct = 0
    overlays_saved = 0

    for label_path in label_files[: args.limit]:
        image_path = _image_path_from_label(label_path)
        if not image_path.exists():
            print(f"Warning: Image not found for {label_path.name}")
            continue

        labels = engine.load_labels(str(image_path))
        if not labels:
            print(f"Warning: No labels for {image_path.name}")
            continue

        filtered_labels, contours = _labels_to_contours(labels)
        if not contours:
            continue

        detections = MockOCREngine.labels_to_detections(filtered_labels)
        matches = match_text_to_contours(
            detections,
            contours,
            max_distance=args.max_distance,
            max_angle_diff=args.max_angle_diff,
        )

        expected_heights = [float(label["elevation"]) for label in filtered_labels]
        correct = sum(
            1
            for idx, expected in enumerate(expected_heights)
            if idx in matches
            and math.isclose(float(matches[idx]), expected, abs_tol=0.01)
        )

        total_labels += len(expected_heights)
        total_matched += len(matches)
        total_correct += correct

        if args.verbose:
            print(
                f"{image_path.name}: {correct}/{len(expected_heights)} correct, "
                f"{len(matches)} matched"
            )

        if (
            args.output_dir
            and overlays_saved < args.max_overlays
            and correct < len(expected_heights)
        ):
            output_path = args.output_dir / f"{image_path.stem}_matching.png"
            _save_overlay(image_path, filtered_labels, contours, matches, output_path)
            overlays_saved += 1

    if total_labels == 0:
        print("No labeled contours found to evaluate.")
        return 1

    accuracy = total_correct / total_labels * 100
    match_rate = total_matched / total_labels * 100
    print(
        f"Matched {total_matched}/{total_labels} labels ({match_rate:.1f}% coverage)."
    )
    print(f"Correct matches: {total_correct}/{total_labels} ({accuracy:.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
