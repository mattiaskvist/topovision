"""Run an end-to-end TopoVision demo with EasyOCR + U-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contour.engine.unet_contour_engine import UNetContourEngine  # noqa: E402
from height_extraction.pipeline import HeightExtractionPipeline  # noqa: E402
from OCR.engine.easyocr_engine import EasyOCREngine  # noqa: E402

DEFAULT_IMAGE = ROOT / "data" / "training" / "N60E013" / "N60E013" / "N60E013_0000.png"


def _resolve_paths(image_arg: str | None, mask_arg: str | None) -> tuple[Path, Path]:
    image_path = Path(image_arg).expanduser().resolve() if image_arg else DEFAULT_IMAGE
    if not image_path.exists():
        msg = (
            f"Image not found: {image_path}\n"
            "Tip: generate tiles with "
            "`uv run python src/data_pipeline/process_data.py "
            "--input data/N60E013/N60E013.shp --output data/training` "
            "and pass --image/--mask explicitly."
        )
        raise FileNotFoundError(msg)

    if mask_arg:
        mask_path = Path(mask_arg).expanduser().resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
    else:
        mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
        if not mask_path.exists():
            print(
                "Mask not found for the selected image. "
                "UNetContourEngine does not require a mask, "
                "so continuing without one. "
                "Provide --mask to use a specific mask file."
            )
            mask_path = image_path

    return image_path, mask_path


def main() -> int:
    """Run the demo."""
    parser = argparse.ArgumentParser(
        description="Run the TopoVision end-to-end demo (EasyOCR + U-Net)."
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to contour mask (optional when using U-Net).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/demo",
        help="Directory to write demo outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to a U-Net checkpoint (.pt).",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="mattiaskvist/topovision-unet",
        help="Hugging Face repo ID for default weights.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="unet/best_model.pt",
        help="Filename in the Hugging Face repo.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for contour prediction.",
    )
    parser.add_argument(
        "--render-mesh",
        action="store_true",
        help="Render the generated mesh interactively with Open3D.",
    )
    args = parser.parse_args()

    image_path, mask_path = _resolve_paths(args.image, args.mask)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contour_engine = UNetContourEngine(
        model_path=args.model_path,
        hf_repo_id=args.hf_repo_id,
        hf_filename=args.hf_filename,
        device=args.device,
        threshold=args.threshold,
    )
    ocr_engine = EasyOCREngine(scale_factors=[2.0, 2.5, 3.0])
    pipeline = HeightExtractionPipeline(
        ocr_engine=ocr_engine,
        contour_engine=contour_engine,
    )

    print(f"Running demo on: {image_path}")
    result = pipeline.run(str(image_path), str(mask_path))

    predicted_mask = contour_engine.predict_mask(str(image_path))
    predicted_mask_path = output_dir / f"{image_path.stem}_predicted_mask.png"
    cv2.imwrite(str(predicted_mask_path), predicted_mask)
    print(f"Saved predicted mask to {predicted_mask_path}")

    viz_path = output_dir / f"{image_path.stem}_result.png"
    pipeline.visualize(result, str(viz_path))

    mesh_path = output_dir / f"{image_path.stem}_mesh.obj"
    pipeline.generate_mesh(result, str(mesh_path))
    if args.render_mesh:
        pipeline.display_mesh(str(mesh_path))

    total = len(result.contours)
    known = sum(1 for c in result.contours if c.source == "ocr")
    inferred = sum(1 for c in result.contours if c.source == "inference")
    print(f"Summary: {total} contours, {known} from OCR, {inferred} inferred.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
