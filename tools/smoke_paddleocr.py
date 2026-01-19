from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


# Make "src/" importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from OCR.engine.paddleocr_engine import PaddleOCREngine


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run python tools/smoke_paddleocr.py /path/to/image.png")
        return 2

    image_path = Path(sys.argv[1]).expanduser().resolve()
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return 2

    img = Image.open(image_path)
    w, h = img.size

    print("Creating engine...")
    engine = PaddleOCREngine()

    out_dir = Path("outputs/paddle_native_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(engine.ocr, "predict"):
        print(f"Saving native PaddleOCR overlay into: {out_dir}")
        native_results = engine.ocr.predict(
            input=str(image_path)
        )  # note: keyword input
        for r in native_results:
            if hasattr(r, "save_to_img"):
                r.save_to_img(save_path=str(out_dir))
            else:
                print("Warning: predict returned an item without save_to_img")
    else:
        print("No predict() found: stai usando l'API ocr() (PaddleOCR 2.x style)")

    print(f"Running OCR on: {image_path}")
    results = engine.extract_with_polygons(str(image_path))

    print(f"Image size: {(w, h)}")
    print(f"Detections: {len(results)}")

    for i, det in enumerate(results[:10]):
        pts = det.polygon.points

        xs = [p[0] for p in pts] if pts else []
        ys = [p[1] for p in pts] if pts else []

        print(f"\n#{i}")
        print(f"  text: {det.text!r}")
        print(f"  confidence: {det.confidence}")
        print(f"  polygon points: {pts}")

        if xs and ys:
            print(
                f"  polygon bounds: x=({min(xs):.1f},{max(xs):.1f}) y=({min(ys):.1f},{max(ys):.1f})"
            )
            if min(xs) < -2 or min(ys) < -2 or max(xs) > w + 2 or max(ys) > h + 2:
                print("  WARNING: polygon outside image bounds")

        if not (0.0 <= float(det.confidence) <= 1.0):
            raise ValueError(f"Invalid confidence: {det.confidence}")

        if not hasattr(pts, "__len__") or len(pts) < 4:
            raise ValueError(f"Polygon has too few points: {pts}")

    print("\nSmoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
