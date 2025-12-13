# Height Extraction Module

This module extracts height curves from topographical maps by combining OCR results with contour line extraction.

## Components

1.  **Contour Extraction** (`src/contour/`):

    - **Abstraction**: `ContourExtractionEngine` (ABC) allows swapping extraction methods.
    - **Implementation**: `CV2ContourEngine` uses `cv2.findContours` and morphological closing.
    - **Extensibility**: To use a different method (e.g., U-Net), implement a new class inheriting from `ContourExtractionEngine`.

2.  **Matching** (`matcher.py`):

    - Matches OCR text detections to the nearest contour line.
    - Uses Euclidean distance from the text centroid to the polyline segments.
    - **Orientation Check**: Verifies that the text rotation aligns with the local tangent of the contour line. Matches are discarded if the angle difference exceeds 30 degrees.

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

### Output Structure

The pipeline now returns a `HeightExtractionOutput` Pydantic model, defined in `schemas.py`.

```python
class ContourLine(BaseModel):
    id: int
    points: list[tuple[int, int]]
    height: float | None
    source: Literal["ocr", "inference", "unknown"]

class HeightExtractionOutput(BaseModel):
    image_path: str
    contours: list[ContourLine]
```

### Custom Contour Engine

To use a custom contour extraction method (e.g., a Deep Learning model):

1.  Create a new class in `src/contour/engine/` that inherits from `ContourExtractionEngine`.
2.  Implement the `extract_contours` method.
3.  Pass an instance of your engine to the `HeightExtractionPipeline` constructor:

```python
from contour.engine.my_custom_engine import MyCustomEngine

contour_engine = MyCustomEngine()
pipeline = HeightExtractionPipeline(contour_engine=contour_engine)
```

## Verification Results

The pipeline has been tested on multiple synthetic images (`data/synthetic/perlin_noise/`).

**Example Run Output (sparse_3_image.png):**

```text
--- Processing sparse_3_image.png ---
Processing .../data/synthetic/perlin_noise/sparse_3_image.png...
Running OCR...
Found 13 text detections.
Extracting contours...
Extracted 17 contours.
Matching text to contours...
Matched 13 contours to heights.
Inferring missing heights...
Adjacency graph has 20 edges.
Estimated Contour Interval: 75.0
Inferred heights for 17 contours (total).
Saved visualization to .../output/height_extraction/sparse_3_result.png
Summary: 17 contours, 13 from OCR, 4 inferred.
```

The pipeline successfully processes all 5 sparse synthetic images, correctly identifying contours, matching OCR text, and inferring missing heights where possible.
