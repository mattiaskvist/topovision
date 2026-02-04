# Data Source

## Download Data

1. Download contour data from [OpenDEM SRTM Download](https://www.opendem.info/srtm_download_contours/)
2. Place the downloaded `.shp` or `.geojson` file in `data/<TILE_NAME>/` directory
3. Use `tile_generator.py` to generate training tiles

## Generate Training Tiles

### Basic Usage (Semantic Segmentation - UNet)

```bash
uv run python src/data_generation/tile_generator.py \
  --input data/N60E014/N60E014.shp \
  --output data/training
```

### Instance Segmentation (Mask2Former)

Generate tiles with instance masks for Mask2Former training:

```bash
uv run python src/data_generation/tile_generator.py \
  --input data/N60E014/N60E014.shp \
  --output data/training \
  --instance-mask
```

### Multiple Input Files

```bash
uv run python src/data_generation/tile_generator.py \
  --input data/N60E012/N60E012.shp data/N60E013/N60E013.shp data/N60E014/N60E014.shp \
  --output data/training \
  --instance-mask
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | `data/N60E014/N60E014.shp` | Input .shp or .geojson file(s) |
| `--output`, `-o` | `data/training` | Output directory |
| `--size` | 512 | Tile size (square) |
| `--dpi` | 150 | Render DPI |
| `--no-mask` | False | Disable binary mask generation |
| `--instance-mask` | False | Generate instance segmentation masks |
| `--mask-thickness` | 2 | Line thickness for masks |

## Convert to COCO Format (for Mask2Former)

After generating tiles with `--instance-mask`, convert to COCO format:

```python
from data_generation.coco_exporter import convert_tiles_to_coco

convert_tiles_to_coco(
    tiles_dir="data/training/N60E014",
    output_path="data/training/N60E014/coco_annotations.json"
)
```

Or use the CLI:

```bash
uv run python -c "from data_generation.coco_exporter import convert_tiles_to_coco; convert_tiles_to_coco('data/training/N60E014', 'data/training/N60E014/coco_annotations.json')"
```

## Output Structure

```
data/training/
├── N60E014/
│   ├── N60E014_0000.png           # RGB tile image
│   ├── N60E014_0000_mask.png      # Binary segmentation mask
│   ├── N60E014_0000_instance.png  # Instance mask (if --instance-mask)
│   ├── N60E014_0000_labels.json   # Tile metadata
│   ├── ...
│   └── coco_annotations.json      # COCO format (after conversion)
```
