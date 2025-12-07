"""Visualizes OCR results with bounding boxes."""

import os

from engine.easyocr_engine import EasyOCREngine
from engine.ocr_engine import OCREngine
from engine.tesseract_engine import TesseractEngine
from PIL import Image, ImageDraw, ImageFont


class OCRVisualizer:
    """Visualizes OCR results with bounding boxes."""

    def __init__(self, engine: OCREngine):
        """Initializes the OCRVisualizer with an OCR engine."""
        self.engine = engine

    def process_and_annotate(self, image_path, output_filename, rotations=None):
        """Processes the image and annotates it with bounding boxes."""
        print(f"--- Processing with {self.engine.__class__.__name__} ---")

        # 1. Get detections (Text + 4 Points)
        # Check if engine supports rotation (EasyOCR) or not (Tesseract)
        if hasattr(self.engine, "extract_with_polygons") and isinstance(
            self.engine, EasyOCREngine
        ):
            rotations = rotations or [90, 180, 270]
            detections = self.engine.extract_with_polygons(
                image_path, rotations=rotations
            )
        else:
            detections = self.engine.extract_with_polygons(image_path)

        # 2. Setup Image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # 3. Draw Polygons
        for text, polygon in detections:
            # polygon is [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

            # Draw the tight polygon/box
            draw.polygon(polygon, outline="lime", width=3)

            # Find the top-left-most point to place the text label
            # We sort points by Y, then X to find the "top" anchor
            top_point = sorted(polygon, key=lambda p: (p[1], p[0]))[0]

            # Draw text label background
            text_w = draw.textlength(text, font=font)
            text_h = 20
            # A small rectangle for the text background
            draw.rectangle(
                [
                    top_point[0],
                    top_point[1] - text_h,
                    top_point[0] + text_w,
                    top_point[1],
                ],
                fill="lime",
            )
            draw.text(
                (top_point[0], top_point[1] - text_h), text, fill="black", font=font
            )

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        img.save(output_filename)
        print(f"Saved: {output_filename}")


if __name__ == "__main__":
    # Ensure you have an image named 'sample.png' in the folder
    from pathlib import Path

    # Ensure you have an image named 'sample.png' in the folder
    # Use pathlib to find the data directory relative to this file
    current_dir = Path(__file__).parent
    # Go up 2 levels from src/OCR/visualizer.py to project root, then to data
    project_root = current_dir.parent.parent
    image_dir = project_root / "data" / "synthetic" / "perlin_noise"
    image_suffix = "_image.png"

    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        exit(1)

    input_images = [f.name for f in image_dir.glob(f"*{image_suffix}")]
    for img in input_images:
        input_img = str(image_dir / img)
        try:
            # Test Tesseract
            viz = OCRVisualizer(TesseractEngine())
            viz.process_and_annotate(
                image_path=input_img, output_filename=f"output/tesseract/{img}"
            )

            # Test EasyOCR
            viz = OCRVisualizer(EasyOCREngine())
            viz.process_and_annotate(
                image_path=input_img, output_filename=f"output/easyocr/{img}"
            )

        except FileNotFoundError:
            print(f"Error: Image {input_img} not found.")
