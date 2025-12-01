# TopoVision

KTH AI Society Research project about extracting topographical data using computer vision

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to automatically check and format code before commits.

**Install hooks:**

```bash
uv run pre-commit install
```

**Run hooks manually:**

```bash
uv run pre-commit run --all-files
```

The hooks will automatically run `ruff check` and `ruff format` on all Python files and Jupyter notebooks before each commit.

## Synthetic Data Generation

This project includes a tool to generate synthetic contour maps with rotated text annotations, useful for training OCR and segmentation models.

**Generate data:**

```bash
uv run python src/data_generation/perlin_noise_generator.py
```

**Output:**

The generated data will be saved in `data/synthetic/perlin_noise`.
The output includes:
- **Images:** Synthetic contour maps with text labels.
- **Masks:** Segmentation masks showing only the contour lines without text labels (images ending in `_mask.png`), suitable for training segmentation models.
- **Debug Images:** Visualizations of the bounding boxes and polygons.
- **Annotations:** A `coco_annotations.json` file containing the annotations in COCO format (used for OCR training).
