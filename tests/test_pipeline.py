from unittest.mock import MagicMock, patch

import numpy as np

from height_extraction.pipeline import HeightExtractionPipeline
from height_extraction.schemas import ContourLine, HeightExtractionOutput
from OCR.engine.ocr_engine import DetectionResult, Polygon


def test_pipeline_run():
    # Mock engines
    mock_ocr = MagicMock()
    mock_ocr.extract_with_polygons.return_value = []  # No text

    mock_contour = MagicMock()
    mock_contour.extract_contours.return_value = []  # No contours

    pipeline = HeightExtractionPipeline(
        ocr_engine=mock_ocr, contour_engine=mock_contour
    )

    # Mock image read failure, then mask read success
    mock_img = MagicMock()
    mock_img.shape = (100, 100, 3)

    with patch("cv2.imread", side_effect=[None, mock_img]):
        # We also need to patch build_adjacency_graph and infer_missing_heights
        # because they might fail with empty inputs if not robust,
        # or we just rely on their robustness (which we tested separately).
        # Let's run it fully to test integration.

        result = pipeline.run("img.png", "mask.png")

        assert len(result.contours) == 0
        assert result.image_path == "img.png"


def test_pipeline_integration_flow():
    # Test with some mocked data
    mock_ocr = MagicMock()
    # Mock a text detection: "100" at (10, 10)
    # Polygon points: top-left, top-right, bottom-right, bottom-left
    # Let's make it horizontal: (10, 10), (30, 10), (30, 20), (10, 20)
    # Centroid: (20, 15)
    detection = DetectionResult(
        text="100",
        polygon=Polygon(points=[(10, 10), (30, 10), (30, 20), (10, 20)]),
        confidence=0.9,
    )
    mock_ocr.extract_with_polygons.return_value = [detection]

    mock_contour = MagicMock()
    # Mock a contour close to the text
    # A horizontal line at y=15 passing through x=20
    # Contour shape: (N, 1, 2)
    # Points: (0, 15), (40, 15)
    contour = np.array([[[0, 15]], [[40, 15]]], dtype=np.int32)
    mock_contour.extract_contours.return_value = [contour]

    pipeline = HeightExtractionPipeline(
        ocr_engine=mock_ocr, contour_engine=mock_contour
    )

    # Mock image read
    mock_img = MagicMock()
    mock_img.shape = (100, 100, 3)

    with patch("cv2.imread", return_value=mock_img):
        result = pipeline.run("img.png", "mask.png")

        assert len(result.contours) == 1
        assert result.contours[0].height == 100.0
        assert result.contours[0].source == "ocr"


def test_visualize_does_not_close_open_contours(tmp_path):
    pipeline = HeightExtractionPipeline(
        ocr_engine=MagicMock(), contour_engine=MagicMock()
    )
    output = HeightExtractionOutput(
        image_path="img.png",
        contours=[
            ContourLine(
                id=0,
                points=[(10, 10), (30, 10), (50, 10)],
                height=100.0,
                source="ocr",
            )
        ],
    )

    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with (
        patch("cv2.imread", return_value=mock_img),
        patch("cv2.imwrite", return_value=True),
        patch("cv2.polylines") as mock_polylines,
    ):
        pipeline.visualize(output, str(tmp_path / "out.png"))

    assert mock_polylines.call_count == 1
    assert mock_polylines.call_args.kwargs["isClosed"] is False
