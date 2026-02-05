# Height Extraction Module

This module extracts height curves from topographical maps by combining OCR results with contour line extraction.

## Components

1. **Contour Extraction** (`src/contour/`):

    - **Abstraction**: `ContourExtractionEngine` (ABC) allows swapping extraction methods.
    - **Implementations**:
      - `CV2ContourEngine`: Uses `cv2.findContours` and morphological closing on binary masks.
      - `UNetContourEngine`: Uses a trained U-Net segmentation model to predict contour masks from input images. Supports loading weights from Hugging Face Hub.
    - **Extensibility**: To use a different method, implement a new class inheriting from `ContourExtractionEngine`.

2. **OCR Engines** (`src/OCR/engine/`):

    - **Abstraction**: `OCREngine` (ABC) allows swapping OCR methods.
    - **Implementations**:
      - `EasyOCREngine`: Production OCR using EasyOCR library.
      - `PaddleOCREngine`: Alternative OCR using PaddleOCR.
      - `MockOCREngine`: Uses per-image `*_labels.json` annotations for testing.

3. **Matching** (`matcher.py`):

    - Matches OCR text detections to the nearest contour line.
    - Uses Euclidean distance from the text centroid to the polyline segments.
    - **Orientation Check**: Verifies that the text rotation aligns with the local tangent of the contour line. Matches are discarded if the angle difference exceeds 30 degrees.

4. **Inference** (`inference.py`):

    - Infers heights for unlabeled contours using **Spatial Gradient Inference**.
    - Builds an adjacency graph of contours.
    - Propagates heights from known to unknown contours by analyzing the spatial direction (gradient) between neighbors.
    - If contour $A$ and $B$ are known, and $C$ is "downstream" of the vector $A \to B$, $C$'s height is inferred by extending the gradient.

5. **Pipeline** (`pipeline.py`):
    - Orchestrates the full process: OCR -> Contours -> Matching -> Inference -> Visualization.

## Usage

### Running the Pipeline

To run the pipeline on training data, execute the following command from the project root:

```bash
cd src
uv run python -m height_extraction.pipeline
```

### Configuration Examples

#### Using U-Net with Hugging Face Hub

```python
from contour.engine.unet_contour_engine import UNetContourEngine
from OCR.engine.easyocr_engine import EasyOCREngine
from height_extraction.pipeline import HeightExtractionPipeline

# Load U-Net model from Hugging Face Hub
contour_engine = UNetContourEngine(
    hf_repo_id="mattiaskvist/topovision-unet",
    hf_filename="best_model.pt",
    device="cuda",  # or "mps" for Mac, "cpu" for fallback
)

ocr_engine = EasyOCREngine()

pipeline = HeightExtractionPipeline(
    ocr_engine=ocr_engine,
    contour_engine=contour_engine,
)

result = pipeline.run(image_path, mask_path)
```

#### Using U-Net with Local Weights

```python
contour_engine = UNetContourEngine(
    model_path="/path/to/best_model.pt",
    device="cpu",
)
```

#### Using CV2 Contour Engine (requires pre-computed mask)

```python
from contour.engine.cv2_contour_engine import CV2ContourEngine

contour_engine = CV2ContourEngine()
pipeline = HeightExtractionPipeline(contour_engine=contour_engine)
```

### Parameters

- `drop_ratio`: In `pipeline.py`, you can set `drop_ratio` (e.g., `0.2`) to simulate missing OCR labels and test the inference logic.

### Verifying Matching with Ground Truth Labels

```bash
uv run python tools/verify_matching.py data/training/N60E013/N60E013 --limit 20
```

### Output Structure

The pipeline returns a `HeightExtractionOutput` Pydantic model, defined in `schemas.py`.

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
