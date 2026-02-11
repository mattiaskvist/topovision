"""Validate mesh generation with perfect and missing heights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contour.engine.unet_contour_engine import UNetContourEngine  # noqa: E402
from height_extraction.pipeline import HeightExtractionPipeline  # noqa: E402
from OCR.engine.mock_ocr_engine import MockOCREngine  # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data" / "training" / "N60E012" / "N60E012"


def _collect_images(data_dir: Path) -> list[Path]:
    images = sorted(data_dir.glob("*.png"))
    return [
        path
        for path in images
        if not path.name.endswith("_mask.png")
        and not path.name.endswith("_instance_mask.png")
    ]


def _mask_for_image(image_path: Path) -> Path | None:
    mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
    return mask_path if mask_path.exists() else None


def main() -> int:
    """Run mesh validation on a batch of images."""
    parser = argparse.ArgumentParser(
        description="Validate mesh generation with perfect and missing heights."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Directory containing training tiles and *_labels.json files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of images to process.",
    )
    parser.add_argument(
        "--drop-ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.4],
        help="OCR drop ratios to simulate missing heights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "mesh_validation",
        help="Directory to save meshes and visualizations.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 2

    images = _collect_images(data_dir)
    if not images:
        print(f"No images found in {data_dir}")
        return 2

    images = images[: args.limit]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_engine = MockOCREngine(data_dir)
    contour_engine = UNetContourEngine()
    pipeline = HeightExtractionPipeline(
        ocr_engine=ocr_engine,
        contour_engine=contour_engine,
    )

    for drop_ratio in args.drop_ratios:
        drop_tag = f"drop_{drop_ratio:.2f}".replace(".", "p")
        run_dir = output_dir / drop_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            mask_path = _mask_for_image(image_path)
            if mask_path is None:
                print(f"Mask not found for {image_path.name}, skipping.")
                continue

            print(f"\n--- {image_path.name} (drop_ratio={drop_ratio:.2f}) ---")
            result = pipeline.run(
                str(image_path), str(mask_path), drop_ratio=drop_ratio
            )

            viz_path = run_dir / f"{image_path.stem}_result.png"
            pipeline.visualize(result, str(viz_path))

            mesh_path = run_dir / f"{image_path.stem}_mesh.obj"
            pipeline.generate_mesh(result, str(mesh_path))

            total = len(result.contours)
            known = sum(1 for c in result.contours if c.source == "ocr")
            inferred = sum(1 for c in result.contours if c.source == "inference")
            unknown = total - known - inferred
            print(
                f"Summary: {total} contours, {known} from OCR, "
                f"{inferred} inferred, {unknown} unknown."
            )

    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
