# Contour Line Segmentation Training

This module provides training infrastructure for a U-Net model that extracts contour lines from topographic map images.

## Overview

The U-Net learns to segment contour lines from map images that contain:

- Terrain backgrounds (colors, patterns)
- Elevation text labels
- Contour lines of varying styles

After training, the model can be used via `UNetContourEngine` to extract individual contour lines from new map images.

## Installation

Install the training dependencies using uv:

```bash
uv sync --group training
```

Or install just the training extras:

```bash
uv pip install torch torchvision segmentation-models-pytorch albumentations tensorboard
```

## Data Generation

First, generate training data with binary masks from real GIS data:

```bash
# Generate tiles + masks from a shapefile
uv run python -m src.dataVisualization.processData \
    --input data/N60E014/N60E014.shp \
    --output data/training \
    --size 512 \
    --mask-thickness 2

# To disable mask generation (images only)
uv run python -m src.dataVisualization.processData \
    --input data/N60E014/N60E014.shp \
    --output data/training \
    --no-mask
```

This generates:

- `{tile_id}.png` — Rendered map tile with terrain, contours, and text labels
- `{tile_id}_mask.png` — Binary mask (white contours on black background)
- `{tile_id}_labels.json` — Label annotations with bounding boxes

## Training

### Basic Training

```bash
uv run python -m src.training.train \
    --data-dir data/training \
    --epochs 100 \
    --device cuda
```

### Full Options

```bash
uv run python -m src.training.train \
    --data-dir data/training \
    --output-dir models \
    --epochs 100 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --encoder resnet34 \
    --device cuda \
    --val-split 0.15 \
    --num-workers 4
```

### Available Encoders

The model uses a pretrained encoder backbone. Options:

- `resnet18` — Faster, smaller
- `resnet34` — Good balance (default)
- `resnet50` — More capacity
- `efficientnet-b0` — Efficient
- `efficientnet-b2` — Higher capacity efficient

### Device Options

- `cuda` — NVIDIA GPU (recommended)
- `mps` — Apple Silicon GPU
- `cpu` — CPU only (slow)

## Cloud GPU Training with Modal

For faster training, use Modal to run on cloud GPUs (NVIDIA T4).

### Setup

```bash
# Install Modal
uv sync --group cloud

# Authenticate (one-time)
uv run modal token new
```

### Upload Training Data

Upload your training data to a persistent Modal volume (run once):

```bash
modal run src/training/modal_train.py::upload_data_local --local-path data/training/N60E014
```

### Run Training

```bash
# Basic training (100 epochs, batch size 8, resnet34 encoder)
modal run src/training/modal_train.py

# With custom options
modal run src/training/modal_train.py --epochs 150 --batch-size 16 --encoder resnet50
```

Training runs on a T4 GPU with a 4-hour timeout. Progress is printed to the console.

### Download Results

After training completes, download the models and TensorBoard logs:

```bash
# List available runs
modal run src/training/modal_train.py::list_runs

# Download a specific run
modal run src/training/modal_train.py::download_results --run-name run_20260128_150000
```

Results are saved to `models/{run_name}/`.

### View TensorBoard Logs

```bash
uv run tensorboard --logdir models/run_20260128_150000/tensorboard
```

## Monitoring Training

Training logs are saved to TensorBoard:

```bash
uv run tensorboard --logdir models
```

Then open <http://localhost:6006> in your browser.

## Output Structure

Training creates:

```
models/
  run_20260128_143022/
    best_model.pt          # Best checkpoint (highest val Dice)
    latest_model.pt        # Most recent checkpoint
    config.json            # Training configuration
    events.out.tfevents.*  # TensorBoard logs
```

## Using the Trained Model

### In Python

```python
from src.contour.engine import UNetContourEngine

# Load trained model
engine = UNetContourEngine(
    model_path="models/run_20260128_143022/best_model.pt",
    device="cuda",
    threshold=0.5,
)

# Extract contours from a map image
contours = engine.extract_contours("path/to/map_image.png")

# Get the predicted mask (for visualization)
mask = engine.predict_mask("path/to/map_image.png")
```

### In the Height Extraction Pipeline

```python
from src.contour.engine import UNetContourEngine
from src.height_extraction.pipeline import HeightExtractionPipeline

# Use U-Net instead of CV2 for contour extraction
contour_engine = UNetContourEngine("models/run_xxx/best_model.pt")

pipeline = HeightExtractionPipeline(
    contour_engine=contour_engine,
    # ... other options
)

result = pipeline.run(image_path="map.png")
```

## Configuration

Training configuration is defined in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `data/dataVisualization/output` | Training data directory |
| `output_dir` | `models` | Output directory for checkpoints |
| `image_size` | 512 | Input image size |
| `batch_size` | 8 | Training batch size |
| `num_epochs` | 100 | Number of training epochs |
| `learning_rate` | 1e-4 | Initial learning rate |
| `encoder_name` | `resnet34` | Backbone encoder |
| `encoder_weights` | `imagenet` | Pretrained weights |
| `val_split` | 0.15 | Validation split fraction |
| `dice_weight` | 1.0 | Dice loss weight |
| `bce_weight` | 1.0 | BCE loss weight |

## Data Augmentations

Training augmentations (applied on-the-fly):

- Horizontal/vertical flip
- Random rotation (90°)
- Shift, scale, rotate
- Elastic deformation
- Brightness/contrast jitter
- Gaussian noise
- Blur

Validation uses only resize and normalization (no augmentations).

## Tips

1. **Start with ~500-1000 tiles** for initial experiments
2. **Monitor validation Dice score** — should reach 0.7+ for good segmentation
3. **Use `resnet34`** as default encoder — good balance of speed/accuracy
4. **Line thickness of 2-3px** in masks works well for training
5. **Check TensorBoard** for loss curves and sample predictions
