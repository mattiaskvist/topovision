# Height Extraction Module

This module extracts height curves from topographical maps by combining OCR results with contour line extraction.

## Components

1.  **Contour Extraction** (`contours.py`):

    - Extracts vector polylines from binary masks using `cv2.findContours`.
    - Applies morphological closing to merge close parallel lines (often caused by thick raster lines).
    - Simplifies lines using `cv2.approxPolyDP`.

2.  **Matching** (`matcher.py`):

    - Matches OCR text detections to the nearest contour line.
    - Uses Euclidean distance from the text centroid to the polyline segments.

3.  **Inference** (`inference.py`):

    - Infers heights for unlabeled contours using **Spatial Gradient Inference**.
    - Builds an adjacency graph of contours.
    - Propagates heights from known to unknown contours by analyzing the spatial direction (gradient) between neighbors.
    - If contour $A$ and $B$ are known, and $C$ is "downstream" of the vector $A \to B$, $C$'s height is inferred by extending the gradient.

4.  **Pipeline** (`pipeline.py`):
    - Orchestrates the full process: OCR -> Contours -> Matching -> Inference -> Visualization.

## Usage

### Running the Pipeline

To run the pipeline on the synthetic data, execute the following command from the project root:

```bash
cd src
uv run python -m height_extraction.pipeline
```

### Parameters

- `drop_ratio`: In `pipeline.py`, you can set `drop_ratio` (e.g., `0.2`) to simulate missing OCR labels and test the inference logic.

### Mock OCR

Currently, the system uses a `MockOCREngine` that reads ground truth annotations from `data/synthetic/perlin_noise/coco_annotations.json`. To switch to a real OCR engine (like PaddleOCR), modify `pipeline.py` to initialize `HeightExtractionPipeline` with `PaddleOCREngine()`.
