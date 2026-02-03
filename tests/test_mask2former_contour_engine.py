"""Tests for Mask2FormerContourEngine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from contour.engine.mask2former_engine import Mask2FormerContourEngine


@pytest.fixture
def mock_processor():
    """Create a mock Mask2Former image processor."""
    processor = MagicMock()
    # Mock processing returns dict with pixel_values tensor
    processor.return_value = {
        "pixel_values": torch.zeros((1, 3, 512, 512)),
        "pixel_mask": torch.ones((1, 512, 512)),
    }
    return processor


@pytest.fixture
def mock_model():
    """Create a mock Mask2FormerForUniversalSegmentation model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.device = torch.device("cpu")
    # Mock output with instance predictions
    output = MagicMock()
    output.class_queries_logits = torch.zeros((1, 100, 2))  # 100 queries, 2 classes
    output.masks_queries_logits = torch.zeros((1, 100, 512, 512))  # 100 masks
    model.return_value = output
    return model


@pytest.fixture
def mock_processor_class(mock_processor):
    """Create a mock for the processor class."""
    return MagicMock(from_pretrained=MagicMock(return_value=mock_processor))


@pytest.fixture
def mock_model_class(mock_model):
    """Create a mock for the model class."""
    return MagicMock(from_pretrained=MagicMock(return_value=mock_model))


def create_engine_with_mocks(mock_model, mock_processor, **kwargs):
    """Helper to create engine with mocked dependencies."""
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")

    with (
        patch(
            "contour.engine.mask2former_engine.Mask2FormerForUniversalSegmentation"
        ) as model_cls,
        patch(
            "contour.engine.mask2former_engine.Mask2FormerImageProcessor"
        ) as proc_cls,
    ):
        model_cls.from_pretrained.return_value = mock_model
        proc_cls.from_pretrained.return_value = mock_processor
        engine = Mask2FormerContourEngine(
            model_path="/fake/model",
            device="cpu",
            **kwargs,
        )
    return engine


# --- Initialization Tests ---


def test_init_with_local_path(mock_model, mock_processor):
    """Test initialization with local model path."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    assert engine.model_path == Path("/fake/model")
    assert engine.device == torch.device("cpu")
    assert engine.score_threshold == 0.5
    assert engine.min_length == 50.0


def test_init_with_hf_model_name(mock_model, mock_processor):
    """Test initialization with Hugging Face model name."""
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")

    with (
        patch(
            "contour.engine.mask2former_engine.Mask2FormerForUniversalSegmentation"
        ) as model_cls,
        patch(
            "contour.engine.mask2former_engine.Mask2FormerImageProcessor"
        ) as proc_cls,
    ):
        model_cls.from_pretrained.return_value = mock_model
        proc_cls.from_pretrained.return_value = mock_processor

        _engine = Mask2FormerContourEngine(
            model_path="facebook/mask2former-swin-tiny-coco-instance",
            device="cpu",
        )

        model_cls.from_pretrained.assert_called_once()
        proc_cls.from_pretrained.assert_called_once()


def test_init_custom_threshold_and_min_length(mock_model, mock_processor):
    """Test initialization with custom threshold and min_length."""
    engine = create_engine_with_mocks(
        mock_model, mock_processor, score_threshold=0.7, min_length=100.0
    )

    assert engine.score_threshold == 0.7
    assert engine.min_length == 100.0


def test_init_custom_epsilon_factor(mock_model, mock_processor):
    """Test initialization with custom epsilon_factor."""
    engine = create_engine_with_mocks(mock_model, mock_processor, epsilon_factor=0.01)

    assert engine.epsilon_factor == 0.01


# --- Device Selection Tests ---


def test_device_fallback_to_cpu(mock_model, mock_processor):
    """Test that device falls back to CPU when CUDA/MPS unavailable."""
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch(
            "contour.engine.mask2former_engine.Mask2FormerForUniversalSegmentation"
        ) as model_cls,
        patch(
            "contour.engine.mask2former_engine.Mask2FormerImageProcessor"
        ) as proc_cls,
    ):
        model_cls.from_pretrained.return_value = mock_model
        proc_cls.from_pretrained.return_value = mock_processor

        engine = Mask2FormerContourEngine(
            model_path="/fake/model",
            device="cuda",
        )

        assert engine.device == torch.device("cpu")


def test_device_uses_cuda_when_available(mock_model, mock_processor):
    """Test that CUDA device is used when available."""
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cuda")

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "contour.engine.mask2former_engine.Mask2FormerForUniversalSegmentation"
        ) as model_cls,
        patch(
            "contour.engine.mask2former_engine.Mask2FormerImageProcessor"
        ) as proc_cls,
    ):
        model_cls.from_pretrained.return_value = mock_model
        proc_cls.from_pretrained.return_value = mock_processor

        engine = Mask2FormerContourEngine(
            model_path="/fake/model",
            device="cuda",
        )

        assert engine.device == torch.device("cuda")


# --- Preprocessing Tests ---


def test_preprocess_image(mock_model, mock_processor):
    """Test image preprocessing for inference."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    # Create a test image
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[100:150, 50:200] = 255  # White rectangle

    result = engine._preprocess_image(image)

    assert "pixel_values" in result
    mock_processor.assert_called_once()


def test_preprocess_image_from_path(mock_model, mock_processor, tmp_path):
    """Test image preprocessing from file path."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    # Create a test image file
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[100:150, 50:200] = 255
    image_path = tmp_path / "test_image.png"
    cv2.imwrite(str(image_path), image)

    # Load and preprocess
    loaded_image = cv2.imread(str(image_path))
    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    result = engine._preprocess_image(loaded_image)

    assert "pixel_values" in result


# --- Instance Prediction Tests ---


def test_predict_instances_returns_list(mock_model, mock_processor):
    """Test that predict_instances returns a list of instance masks."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    # Create mock post-processing result
    mock_result = {
        "segmentation": torch.zeros((256, 256), dtype=torch.int32),
        "segments_info": [
            {"id": 1, "label_id": 0, "score": 0.9},
            {"id": 2, "label_id": 0, "score": 0.8},
        ],
    }

    with (
        patch.object(
            mock_processor,
            "post_process_instance_segmentation",
            return_value=[mock_result],
        ),
        patch.object(engine, "_load_image", return_value=np.zeros((256, 256, 3))),
    ):
        instances = engine.predict_instances("/fake/image.png")

    assert isinstance(instances, list)


def test_predict_instances_filters_by_score(mock_model, mock_processor):
    """Test that instances below score threshold are filtered."""
    engine = create_engine_with_mocks(mock_model, mock_processor, score_threshold=0.85)

    # Create segmentation with 2 instances
    segmentation = torch.zeros((256, 256), dtype=torch.int32)
    segmentation[50:100, 50:100] = 1  # Instance 1
    segmentation[150:200, 150:200] = 2  # Instance 2

    mock_result = {
        "segmentation": segmentation,
        "segments_info": [
            {"id": 1, "label_id": 0, "score": 0.9},  # Above threshold
            {"id": 2, "label_id": 0, "score": 0.7},  # Below threshold
        ],
    }

    with (
        patch.object(
            mock_processor,
            "post_process_instance_segmentation",
            return_value=[mock_result],
        ),
        patch.object(engine, "_load_image", return_value=np.zeros((256, 256, 3))),
    ):
        instances = engine.predict_instances("/fake/image.png")

    # Only one instance should pass the threshold
    assert len(instances) == 1


def test_predict_instances_returns_binary_masks(mock_model, mock_processor):
    """Test that each instance mask is binary (0 or 255)."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    segmentation = torch.zeros((256, 256), dtype=torch.int32)
    segmentation[50:100, 50:100] = 1

    mock_result = {
        "segmentation": segmentation,
        "segments_info": [
            {"id": 1, "label_id": 0, "score": 0.9},
        ],
    }

    with (
        patch.object(
            mock_processor,
            "post_process_instance_segmentation",
            return_value=[mock_result],
        ),
        patch.object(engine, "_load_image", return_value=np.zeros((256, 256, 3))),
    ):
        instances = engine.predict_instances("/fake/image.png")

    assert len(instances) == 1
    mask = instances[0]
    unique_values = np.unique(mask)
    assert all(v in [0, 255] for v in unique_values)


# --- Contour Extraction Tests ---


def test_extract_contours_returns_list_of_arrays(mock_model, mock_processor):
    """Test that extract_contours returns list of numpy arrays."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    # Create mock instance masks
    mask1 = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(mask1, (50, 100), (200, 100), 255, 2)

    with patch.object(engine, "predict_instances", return_value=[mask1]):
        contours = engine.extract_contours("/fake/image.png")

    assert isinstance(contours, list)
    for contour in contours:
        assert isinstance(contour, np.ndarray)


def test_extract_contours_shape_n_1_2(mock_model, mock_processor):
    """Test that each contour has shape (N, 1, 2)."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    # Create a mask with a clear contour
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (200, 200), 255, 2)

    with patch.object(engine, "predict_instances", return_value=[mask]):
        contours = engine.extract_contours("/fake/image.png")

    assert len(contours) > 0
    for contour in contours:
        assert contour.ndim == 3
        assert contour.shape[1] == 1
        assert contour.shape[2] == 2


def test_extract_contours_filters_by_min_length(mock_model, mock_processor):
    """Test that short contours are filtered by min_length."""
    engine = create_engine_with_mocks(mock_model, mock_processor, min_length=100.0)

    # Create a mask with a small contour
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (110, 110), 255, 1)  # Very small

    with patch.object(engine, "predict_instances", return_value=[mask]):
        contours = engine.extract_contours("/fake/image.png")

    # Small contour should be filtered out
    assert len(contours) == 0


def test_extract_contours_from_multiple_instances(mock_model, mock_processor):
    """Test extraction from multiple instance masks."""
    engine = create_engine_with_mocks(mock_model, mock_processor, min_length=10.0)

    # Create two instance masks with clear contours
    mask1 = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask1, (20, 20), (80, 80), 255, 2)

    mask2 = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask2, (120, 120), (200, 200), 255, 2)

    with patch.object(engine, "predict_instances", return_value=[mask1, mask2]):
        contours = engine.extract_contours("/fake/image.png")

    # Should have contours from both instances
    assert len(contours) >= 2


# --- Integration-like Tests ---


def test_extract_contours_with_real_mask_processing(mock_model, mock_processor):
    """Test contour extraction with actual cv2 processing."""
    engine = create_engine_with_mocks(
        mock_model, mock_processor, min_length=20.0, epsilon_factor=0.005
    )

    # Create a mask with a horizontal line contour
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(mask, (30, 128), (220, 128), 255, 3)

    with patch.object(engine, "predict_instances", return_value=[mask]):
        contours = engine.extract_contours("/fake/image.png")

    # Should extract at least one contour
    assert len(contours) > 0

    # Contours should have reasonable coordinates
    for contour in contours:
        assert contour.min() >= 0
        assert contour.max() < 256


def test_extract_contours_empty_when_no_instances(mock_model, mock_processor):
    """Test that empty list is returned when no instances detected."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    with patch.object(engine, "predict_instances", return_value=[]):
        contours = engine.extract_contours("/fake/image.png")

    assert contours == []


# --- Error Handling Tests ---


def test_extract_contours_raises_on_missing_file(mock_model, mock_processor):
    """Test that FileNotFoundError is raised for missing image."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    with pytest.raises(FileNotFoundError):
        engine.extract_contours("/nonexistent/image.png")


def test_predict_instances_with_target_size(mock_model, mock_processor):
    """Test that target_sizes is passed to post-processing."""
    engine = create_engine_with_mocks(mock_model, mock_processor)

    image = np.zeros((480, 640, 3), dtype=np.uint8)  # Non-square image

    mock_result = {
        "segmentation": torch.zeros((480, 640), dtype=torch.int32),
        "segments_info": [],
    }

    with (
        patch.object(
            mock_processor,
            "post_process_instance_segmentation",
            return_value=[mock_result],
        ) as mock_post_process,
        patch.object(engine, "_load_image", return_value=image),
    ):
        engine.predict_instances("/fake/image.png")

    # Verify target_sizes was passed
    call_kwargs = mock_post_process.call_args[1]
    assert "target_sizes" in call_kwargs


# --- Contour Approximation Tests ---


def test_contour_approximation_with_epsilon(mock_model, mock_processor):
    """Test that contour approximation uses epsilon_factor."""
    engine = create_engine_with_mocks(mock_model, mock_processor, epsilon_factor=0.02)

    # Create a detailed mask (circle with many points)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 50, 255, 2)

    with patch.object(engine, "predict_instances", return_value=[mask]):
        contours = engine.extract_contours("/fake/image.png")

    # With high epsilon, contour should have fewer points (more approximated)
    # This is a sanity check that approximation is working
    assert len(contours) > 0


# --- Mask2Former Specific Tests ---


def test_num_labels_configuration(mock_model, mock_processor):
    """Test that num_labels can be configured."""
    engine = create_engine_with_mocks(mock_model, mock_processor, num_labels=2)

    assert engine.num_labels == 2


def test_class_filter_by_label_id(mock_model, mock_processor):
    """Test filtering instances by label ID (class)."""
    engine = create_engine_with_mocks(mock_model, mock_processor, target_label_id=0)

    segmentation = torch.zeros((256, 256), dtype=torch.int32)
    segmentation[50:100, 50:100] = 1
    segmentation[150:200, 150:200] = 2

    mock_result = {
        "segmentation": segmentation,
        "segments_info": [
            {"id": 1, "label_id": 0, "score": 0.9},  # Contour class
            {"id": 2, "label_id": 1, "score": 0.9},  # Other class
        ],
    }

    with (
        patch.object(
            mock_processor,
            "post_process_instance_segmentation",
            return_value=[mock_result],
        ),
        patch.object(engine, "_load_image", return_value=np.zeros((256, 256, 3))),
    ):
        instances = engine.predict_instances("/fake/image.png")

    # Only the contour class instance should be returned
    assert len(instances) == 1
