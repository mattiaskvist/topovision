"""This file tests paddleocr."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Make "src/" importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from OCR.engine.paddleocr_engine import PaddleOCREngine  # noqa: E402


def _as_points(poly) -> list[tuple[int, int]]:
    # poly can be list, tuple, numpy array
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    pts: list[tuple[int, int]] = []
    for p in poly:
        x, y = p
        pts.append((round(x), round(y)))
    return pts


def main() -> int:
    """Run main program."""
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python tools/visualize_paddleocr.py /path/to/image.png [out.png]"  # noqa: E501
        )
        return 2

    image_path = Path(sys.argv[1]).expanduser().resolve()
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return 2

    out_path = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) >= 3
        else (ROOT / "outputs" / f"{image_path.stem}_paddle_annotated.png")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = PaddleOCREngine()
    detections = engine.extract_with_polygons(str(image_path))

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Font: default is fine for quick debug
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        pts = _as_points(det.polygon.points)

        # draw polygon (closed)
        if len(pts) >= 2:
            draw.line([*pts, pts[0]], width=2)

        # label near first point
        x0, y0 = pts[0]
        label = f"{det.text} ({float(det.confidence):.2f})"
        draw.text((x0 + 3, y0 + 3), label, font=font)

    img.save(out_path)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
