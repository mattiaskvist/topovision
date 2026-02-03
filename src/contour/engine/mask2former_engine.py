"""Mask2Former-based contour extraction engine for instance segmentation."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )
except ImportError:
    Mask2FormerForUniversalSegmentation = None
    Mask2FormerImageProcessor = None

from .contour_engine import ContourExtractionEngine


class Mask2FormerContourEngine(ContourExtractionEngine):
    """Extracts contours using Mask2Former for instance segmentation.

    This engine uses a trained Mask2Former model to predict individual
    contour line instances from an input image, then extracts contours
    from each instance mask.

    Unlike semantic segmentation (UNet), instance segmentation can
    distinguish between overlapping or nearby contour lines, providing
    individual contour instances.

    Attributes:
        model_path: Path to the trained model (local or HF Hub name).
        device: Torch device for inference.
        score_threshold: Confidence threshold for instance predictions.
        min_length: Minimum contour length to keep.
        epsilon_factor: Factor for contour approximation.
        num_labels: Number of class labels the model predicts.
        target_label_id: If set, only keep instances of this class.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        score_threshold: float = 0.5,
        min_length: float = 50.0,
        epsilon_factor: float = 0.005,
        num_labels: int = 1,
        target_label_id: int | None = None,
    ):
        """Initialize the Mask2Former contour engine.

        Args:
            model_path: Path to a local model directory or HuggingFace Hub
                model identifier (e.g., "facebook/mask2former-swin-tiny-coco-instance").
            device: Device for inference ("cuda", "mps", or "cpu").
            score_threshold: Confidence threshold for keeping instances.
            min_length: Minimum contour length to keep.
            epsilon_factor: Factor for approximation accuracy.
            num_labels: Number of class labels in the model.
            target_label_id: If specified, only keep instances with this
                label ID. None means keep all classes.
        """
        self.score_threshold = score_threshold
        self.min_length = min_length
        self.epsilon_factor = epsilon_factor
        self.num_labels = num_labels
        self.target_label_id = target_label_id

        # Setup device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Store model path
        self.model_path = model_path

        # If path doesn't exist locally and looks like HF identifier, keep as string
        if not self.model_path.exists():
            self._model_identifier = str(model_path)
        else:
            self._model_identifier = str(self.model_path)

        # Load model and processor
        self.processor, self.model = self._load_model_and_processor()
        self.model.eval()

    def _load_model_and_processor(
        self,
    ) -> tuple[Any, torch.nn.Module]:
        """Load the Mask2Former model and image processor.

        Returns:
            Tuple of (processor, model) ready for inference.
        """
        if Mask2FormerForUniversalSegmentation is None:
            msg = "transformers not installed. Run: pip install transformers"
            raise ImportError(msg)

        processor = Mask2FormerImageProcessor.from_pretrained(self._model_identifier)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self._model_identifier
        )
        model = model.to(self.device)

        return processor, model

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            Image as numpy array (H, W, 3) in RGB format.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _preprocess_image(self, image: np.ndarray) -> dict[str, torch.Tensor]:
        """Preprocess image for model inference.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format.

        Returns:
            Dictionary with preprocessed tensors for the model.
        """
        # Use the HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def predict_instances(self, image_path: str) -> list[np.ndarray]:
        """Predict instance masks from an image.

        Args:
            image_path: Path to the input image.

        Returns:
            List of binary masks, one per detected instance.
            Each mask is a numpy array (H, W) with values 0 or 255.
        """
        # Load image
        image = self._load_image(image_path)
        original_size = image.shape[:2]  # (H, W)

        # Preprocess
        inputs = self._preprocess_image(image)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get instance segmentation
        # Target size is (height, width)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(original_size[0], original_size[1])],
        )[0]

        # Extract individual instance masks
        segmentation = result["segmentation"]  # (H, W) tensor with instance IDs
        segments_info = result["segments_info"]  # List of segment metadata

        instance_masks = []
        for segment in segments_info:
            # Filter by score threshold
            if segment["score"] < self.score_threshold:
                continue

            # Filter by label ID if specified
            if (
                self.target_label_id is not None
                and segment["label_id"] != self.target_label_id
            ):
                continue

            # Extract binary mask for this instance
            instance_id = segment["id"]
            mask = (segmentation == instance_id).numpy().astype(np.uint8) * 255
            instance_masks.append(mask)

        return instance_masks

    def _extract_contour_from_mask(self, mask: np.ndarray) -> list[np.ndarray]:
        """Extract contours from a single binary instance mask.

        Args:
            mask: Binary mask (H, W) with values 0 or 255.

        Returns:
            List of contours from this mask.
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        filtered_contours = []
        for contour in contours:
            # Filter by length
            length = cv2.arcLength(contour, closed=False)
            if length < self.min_length:
                continue

            # Approximate contour to reduce points
            epsilon = self.epsilon_factor * length
            approx = cv2.approxPolyDP(contour, epsilon, closed=False)

            # Ensure shape is (N, 1, 2)
            if approx.ndim == 2:
                approx = approx.reshape(-1, 1, 2)

            filtered_contours.append(approx)

        return filtered_contours

    def extract_contours(self, image_path: str) -> list[np.ndarray]:
        """Extract contours from an image using Mask2Former instance segmentation.

        Args:
            image_path: Path to the input image.

        Returns:
            List of contours, where each contour is a numpy array of shape (N, 1, 2).

        Raises:
            FileNotFoundError: If the image file cannot be read.
        """
        # Get instance masks
        instance_masks = self.predict_instances(image_path)

        if not instance_masks:
            return []

        # Extract contours from each instance mask
        all_contours = []
        for mask in instance_masks:
            contours = self._extract_contour_from_mask(mask)
            all_contours.extend(contours)

        return all_contours
