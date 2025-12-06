# Contour Map OCR Training Dataset Generator

Generate OCR/text detection training datasets from topographic contour maps.

## Purpose

Creates **training pairs** for teaching models to read elevation numbers from topographic contour maps:
- **X (Input)**: Natural terrain-colored contour map images with elevation labels
- **Y (Labels)**: Ground truth JSON with elevation values + bounding boxes

## What It Generates

1. **Contour Images**: Topographic maps with elevation-colored terrain and labeled contour lines
2. **Label Files**: JSON with elevation value + bounding box for each text label
3. **Segmentation Masks**: Binary masks of contour lines (optional)
4. **Metadata**: World coordinates, elevation ranges, dataset statistics

## Quick Start

```python
from dataHelpers import ContourDatasetGenerator

generator = ContourDatasetGenerator(
    output_dir="output",
    tile_size=512,
    grid_size=3,
    dpi=150
)

generator.process_file("data/N63E016.shp")
```

## Output Structure

```
output/N63E016/
├── images/
│   ├── N63E016_r00_c00.png           # Image with labeled contours
│   ├── N63E016_r00_c00_labels.json   # Ground truth labels
│   ├── N63E016_r00_c01.png
│   ├── N63E016_r00_c01_labels.json
│   └── ...
├── masks/
│   └── N63E016_r00_c00_mask.png     # Binary masks (optional)
└── dataset_summary.json              # Complete metadata
```

## Label Format

Each `*_labels.json`:
```json
{
  "image": "N63E016_r00_c00.png",
  "image_size": [256, 256],
  "labels": [
    {
      "elevation": 475,
      "text": "475",
      "bbox_pixels": [73.99, 136.29, 29.13, 14.0],
      "bbox_normalized": [0.289, 0.532, 0.114, 0.055]
    }
  ]
}
```

- `elevation`: Height of contour line in meters
- `text`: String representation
- `bbox_pixels`: [x, y, width, height] in pixels
- `bbox_normalized`: [x, y, width, height] in [0, 1] range

Perfect for YOLO, Faster R-CNN, or custom OCR models.

## CLI Usage

```bash
python -m dataHelpers \
    --input data/ \
    --output output/ \
    --tile-size 512 \
    --grid-size 3 \
    --dpi 150
```

Options:
- `--tile-size`: 50, 256, 512, or 1024 pixels
- `--grid-size`: NxN grid split (e.g., 3 = 9 tiles)
- `--no-masks`: Skip mask generation
- `--dpi`: Image resolution

## Training Your Model

Use the image-label pairs to train:
- **OCR models**: Read elevation numbers
- **Text detection**: Locate text boxes
- **Object detection**: YOLO/Faster R-CNN compatible
- **End-to-end**: Map image → elevation readings

## Data Source

Download SRTM contour shapefiles:
https://www.opendem.info/srtm_download_contours/

## Examples

See `example.py`:
- Quick 3x3 test
- 50x50 pixel tiles
- High-res 1024px datasets
- Batch processing
- Data inspection

## Features

✅ Natural terrain coloring (green valleys → brown mountains → white peaks)  
✅ Elevation labels on contour lines  
✅ Ground truth labels (elevation + bounding box)  
✅ Pixel + normalized coordinates  
✅ World coordinate tracking  
✅ Configurable tile sizes (50-1024px)  
✅ Batch processing  
✅ Complete metadata  

## Assignment Deliverables

✅ Topographic images with natural terrain colors  
✅ Elevation labels on contour lines  
✅ Ground truth labels with bounding boxes  
✅ Metadata tracking each contour's height  
✅ World coordinates for validation  
✅ CLI with options  
✅ Batch generation  
✅ 50x50 tile support  
✅ Documentation  
