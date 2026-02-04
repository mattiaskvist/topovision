"""Tests for UNetContourEngine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from contour.engine.unet_contour_engine import UNetContourEngine


@pytest.fixture
def mock_model():
    """Create a mock U-Net model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    # Return logits that produce a mask with a horizontal line
    # Shape: [1, 1, 512, 512]
    logits = torch.zeros((1, 1, 512, 512))
    logits[:, :, 250:260, 50:450] = 5.0  # High logits = line after sigmoid
    model.return_value = logits
    return model


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint dictionary."""
    return {
        "model_state_dict": {},
        "config": {"encoder_name": "resnet34"},
        "epoch": 50,
        "val_dice": 0.85,
    }


def create_engine_with_mocks(mock_model, mock_checkpoint, **kwargs):
    """Helper to create engine with mocked dependencies."""
    with (
        patch("torch.load", return_value=mock_checkpoint),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model),
    ):
        engine = UNetContourEngine(model_path="/fake/model.pt", device="cpu", **kwargs)
    return engine


# --- Initialization Tests ---


def test_init_with_local_path(mock_model, mock_checkpoint):
    """Test initialization with local model path."""
    with (
        patch("torch.load", return_value=mock_checkpoint),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model),
    ):
        engine = UNetContourEngine(model_path="/fake/model.pt", device="cpu")

        assert engine.model_path == Path("/fake/model.pt")
        assert engine.device == torch.device("cpu")
        assert engine.threshold == 0.5
        assert engine.min_length == 50.0


def test_init_with_hf_hub(mock_model, mock_checkpoint):
    """Test initialization with Hugging Face Hub."""
    mock_hf_download = MagicMock(return_value="/cached/model.pt")

    with (
        patch(
            "contour.engine.unet_contour_engine.UNetContourEngine._download_from_hub",
            mock_hf_download,
        ),
        patch("torch.load", return_value=mock_checkpoint),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model),
    ):
        engine = UNetContourEngine(
            hf_repo_id="test/repo", hf_filename="model.pt", device="cpu"
        )

        mock_hf_download.assert_called_once_with("test/repo", "model.pt")
        assert str(engine.model_path) == "/cached/model.pt"


def test_init_custom_threshold_and_min_length(mock_model, mock_checkpoint):
    """Test initialization with custom threshold and min_length."""
    engine = create_engine_with_mocks(
        mock_model, mock_checkpoint, threshold=0.7, min_length=100.0
    )

    assert engine.threshold == 0.7
    assert engine.min_length == 100.0


def test_init_custom_epsilon_factor(mock_model, mock_checkpoint):
    """Test initialization with custom epsilon_factor."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint, epsilon_factor=0.01)

    assert engine.epsilon_factor == 0.01


# --- Device Selection Tests ---


def test_device_fallback_to_cpu():
    """Test that device falls back to CPU when CUDA/MPS unavailable."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)

    mock_checkpoint = {"model_state_dict": {}, "config": {}}

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.load", return_value=mock_checkpoint),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model),
    ):
        engine = UNetContourEngine(model_path="/fake/model.pt", device="cuda")

        assert engine.device == torch.device("cpu")


def test_device_uses_cuda_when_available():
    """Test that CUDA device is used when available."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)

    mock_checkpoint = {"model_state_dict": {}, "config": {}}

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.load", return_value=mock_checkpoint),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model),
    ):
        engine = UNetContourEngine(model_path="/fake/model.pt", device="cuda")

        assert engine.device == torch.device("cuda")


# --- HuggingFace Hub Tests ---


def test_download_from_hub_success(mock_model, mock_checkpoint):
    """Test successful download from Hugging Face Hub."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    mock_hf_download = MagicMock(return_value="/cached/model.pt")
    mock_hf_module = MagicMock()
    mock_hf_module.hf_hub_download = mock_hf_download

    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = engine._download_from_hub("test/repo", "model.pt")

    mock_hf_download.assert_called_once_with(repo_id="test/repo", filename="model.pt")
    assert result == Path("/cached/model.pt")


# --- Model Loading Tests ---


def test_load_model_with_default_encoder(mock_model):
    """Test model loading uses default encoder when not in checkpoint."""
    checkpoint_no_config = {"model_state_dict": {}}

    with (
        patch("torch.load", return_value=checkpoint_no_config),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model) as mock_unet,
    ):
        UNetContourEngine(model_path="/fake/model.pt", device="cpu")

        mock_unet.assert_called_once_with(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )


def test_load_model_with_custom_encoder(mock_model):
    """Test model loading uses encoder from checkpoint config."""
    checkpoint_custom = {
        "model_state_dict": {},
        "config": {"encoder_name": "efficientnet-b4"},
    }

    with (
        patch("torch.load", return_value=checkpoint_custom),
        patch("segmentation_models_pytorch.Unet", return_value=mock_model) as mock_unet,
    ):
        UNetContourEngine(model_path="/fake/model.pt", device="cpu")

        mock_unet.assert_called_once_with(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )


# --- Preprocessing Tests ---


def test_preprocess_image_normalization(mock_model, mock_checkpoint):
    """Test that preprocessing normalizes image correctly."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    # Create a simple RGB image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128

    tensor, original_size = engine._preprocess_image(image)

    assert original_size == (100, 100)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == torch.float32


def test_preprocess_image_resizes_to_512(mock_model, mock_checkpoint):
    """Test that preprocessing resizes to 512x512."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    # Create various sized images
    for size in [(256, 256), (1024, 768), (100, 200)]:
        image = np.zeros((*size, 3), dtype=np.uint8)
        tensor, original_size = engine._preprocess_image(image)

        assert tensor.shape == (1, 3, 512, 512)
        assert original_size == size


# --- Postprocessing Tests ---


def test_postprocess_mask_thresholding(mock_model, mock_checkpoint):
    """Test that postprocessing applies threshold correctly."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint, threshold=0.5)

    # Create logits with clear separation
    logits = torch.zeros((1, 1, 100, 100))
    logits[:, :, :50, :] = 5.0  # High prob (sigmoid > 0.5)
    logits[:, :, 50:, :] = -5.0  # Low prob (sigmoid < 0.5)

    mask = engine._postprocess_mask(logits, (100, 100))

    assert mask.shape == (100, 100)
    assert np.all(mask[:50, :] == 255)  # Above threshold
    assert np.all(mask[50:, :] == 0)  # Below threshold


def test_postprocess_mask_resizes_to_original(mock_model, mock_checkpoint):
    """Test that postprocessing resizes to original size."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    logits = torch.zeros((1, 1, 512, 512))

    for original_size in [(100, 200), (768, 1024), (256, 256)]:
        mask = engine._postprocess_mask(logits, original_size)
        # Note: cv2.resize takes (width, height), but numpy shape is (height, width)
        assert mask.shape == original_size


# --- Contour Extraction Tests ---


def test_extract_contours_file_not_found(mock_model, mock_checkpoint):
    """Test error when image file not found."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    with (
        patch("cv2.imread", return_value=None),
        pytest.raises(FileNotFoundError, match="Could not read image"),
    ):
        engine.extract_contours("nonexistent.png")


def test_extract_contours_returns_list(mock_model, mock_checkpoint):
    """Test that extract_contours returns a list of contours."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    # Create a test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("cv2.imread", return_value=test_image):
        contours = engine.extract_contours("test.png")

        assert isinstance(contours, list)


def test_extract_contours_from_mask_with_line(mock_model, mock_checkpoint):
    """Test contour extraction from mask with horizontal line."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint, min_length=10.0)

    # Create mask with a thick horizontal line
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(mask, (10, 50), (90, 50), 255, 5)

    contours = engine._extract_contours_from_mask(mask)

    assert len(contours) >= 1
    # Contour should be roughly horizontal line length
    length = cv2.arcLength(contours[0], closed=False)
    assert length > 50


def test_extract_contours_from_mask_empty(mock_model, mock_checkpoint):
    """Test contour extraction from empty mask."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    mask = np.zeros((100, 100), dtype=np.uint8)

    contours = engine._extract_contours_from_mask(mask)

    assert len(contours) == 0


def test_extract_contours_min_length_filtering(mock_model, mock_checkpoint):
    """Test that min_length filters short contours."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint, min_length=50.0)

    # Create mask with one long and one short line
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (10, 50), (150, 50), 255, 3)  # Long line (~140px)
    cv2.line(mask, (10, 100), (30, 100), 255, 3)  # Short line (~20px)

    contours = engine._extract_contours_from_mask(mask)

    # Should only keep the long line
    assert len(contours) == 1


# --- Predict Mask Tests ---


def test_predict_mask_returns_numpy_array(mock_model, mock_checkpoint):
    """Test that predict_mask returns numpy array."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("cv2.imread", return_value=test_image):
        mask = engine.predict_mask("test.png")

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)


def test_predict_mask_file_not_found(mock_model, mock_checkpoint):
    """Test error when image file not found in predict_mask."""
    engine = create_engine_with_mocks(mock_model, mock_checkpoint)

    with (
        patch("cv2.imread", return_value=None),
        pytest.raises(FileNotFoundError, match="Could not read image"),
    ):
        engine.predict_mask("nonexistent.png")
