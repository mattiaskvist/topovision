# Mask2Former Inference

This module provides inference utilities for the Mask2Former contour instance segmentation model.

## Installation

Make sure you have the required dependencies:

```bash
uv sync
```

## Usage

### Command Line

Run inference on an image:

```bash
# Basic usage (uses HuggingFace model by default)
PYTHONPATH=src uv run python src/training/mask2former/inference.py path/to/image.png

# Save visualization output
PYTHONPATH=src uv run python src/training/mask2former/inference.py path/to/image.png --output output/prediction.png

# Use a local model
PYTHONPATH=src uv run python src/training/mask2former/inference.py path/to/image.png \
    --model models/mask2former/mask2former_20260203_191238/best_model_hf \
    --subfolder .

# Adjust confidence threshold
PYTHONPATH=src uv run python src/training/mask2former/inference.py path/to/image.png --threshold 0.7

# Force CPU inference
PYTHONPATH=src uv run python src/training/mask2former/inference.py path/to/image.png --device cpu
```

### Python API

```python
from training.mask2former.inference import (
    load_model,
    predict,
    predict_with_skeletons,
    visualize_predictions,
    visualize_skeletons,
)
import cv2

# Load model
model, processor, device = load_model()

# Load image (must be RGB)
image = cv2.imread("path/to/image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Basic prediction (instance masks)
predictions = predict(model, processor, image, device, threshold=0.5)
print(f"Found {len(predictions['masks'])} contour instances")

# Prediction with skeleton extraction
predictions = predict_with_skeletons(
    model, processor, image, device,
    threshold=0.5,
    connect_gaps=True,        # Connect disconnected skeleton segments
    max_gap_distance=15,      # Max pixels to bridge between segments
    smooth_polylines=False,   # Apply spline smoothing
)

# Access results
for i, (mask, skeleton, score) in enumerate(zip(
    predictions["masks"],
    predictions["skeletons"],
    predictions["scores"]
)):
    print(f"Instance {i}: score={score:.3f}, mask shape={mask.shape}")

# Polylines are ordered (x, y) coordinates
for polyline in predictions["polylines"]:
    print(f"Polyline with {len(polyline)} points")

# Visualize
vis_masks = visualize_predictions(image, predictions, alpha=0.5)
vis_skeletons = visualize_skeletons(image, predictions, line_thickness=2)

# Save
cv2.imwrite("masks.png", cv2.cvtColor(vis_masks, cv2.COLOR_RGB2BGR))
cv2.imwrite("skeletons.png", cv2.cvtColor(vis_skeletons, cv2.COLOR_RGB2BGR))
```

## Output Format

The `predict()` function returns a dictionary with:

- `masks`: List of binary masks (H, W) for each detected contour instance
- `scores`: Confidence scores for each instance
- `labels`: Class labels (all contours have the same label)

The `predict_with_skeletons()` function adds:

- `skeletons`: Single-pixel wide skeleton masks
- `polylines`: Ordered (x, y) coordinate arrays for each skeleton

## Skeleton Gap Connection

The skeletonization process can produce disconnected fragments. The `connect_gaps` option (enabled by default) automatically bridges nearby endpoints:

- Finds skeleton endpoints (pixels with only one neighbor)
- Connects pairs within `max_gap_distance` pixels
- Verifies endpoints belong to different components before connecting

## Model Sources

Default model: `mattiaskvist/topovision-segmentation` (HuggingFace Hub)

Local models are saved in `models/mask2former/` with the structure:

```
models/mask2former/mask2former_YYYYMMDD_HHMMSS/
├── best_model.pt          # PyTorch checkpoint
├── best_model_hf/         # HuggingFace format
│   ├── config.json
│   └── model.safetensors
└── tensorboard/           # Training logs
```
