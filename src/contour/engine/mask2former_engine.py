"""Mask2Former-based contour extraction engine for instance segmentation."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

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
        use_skeletonization: If True, use skeletonization instead of findContours.
        connect_skeleton_gaps: Whether to connect disconnected skeleton segments.
        max_gap_distance: Maximum pixel distance to bridge skeleton gaps.
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
        use_skeletonization: bool = False,
        connect_skeleton_gaps: bool = True,
        max_gap_distance: int = 15,
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
            use_skeletonization: If True, extract contours using morphological
                skeletonization (single-pixel wide lines) instead of findContours.
            connect_skeleton_gaps: Whether to connect nearby disconnected
                skeleton segments (only used if use_skeletonization=True).
            max_gap_distance: Maximum pixel distance to bridge between
                disconnected skeleton segments.
        """
        self.score_threshold = score_threshold
        self.min_length = min_length
        self.epsilon_factor = epsilon_factor
        self.num_labels = num_labels
        self.target_label_id = target_label_id
        self.use_skeletonization = use_skeletonization
        self.connect_skeleton_gaps = connect_skeleton_gaps
        self.max_gap_distance = max_gap_distance

        # Setup device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Store model path (convert to Path if string)
        if isinstance(model_path, str):
            self.model_path = Path(model_path)
        else:
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

    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Extract a single-pixel wide skeleton from a binary mask.

        Args:
            mask: Binary mask (H, W) with values 0 or 255.

        Returns:
            Skeleton mask (H, W) with single-pixel wide lines.
        """
        # Skeletonize expects binary image
        binary_mask = mask > 0
        skeleton = skeletonize(binary_mask)
        skeleton = skeleton.astype(np.uint8)

        if self.connect_skeleton_gaps:
            skeleton = self._connect_skeleton_gaps(skeleton)

        return skeleton

    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """Find endpoint pixels in a skeleton (pixels with only one neighbor).

        Args:
            skeleton: Binary skeleton image.

        Returns:
            Array of endpoint coordinates as (N, 2) in (y, x) format.
        """
        # Convolve with a kernel that counts 8-connected neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)

        # Endpoints have exactly 1 neighbor and are part of the skeleton
        endpoints = (neighbor_count == 1) & (skeleton > 0)
        return np.column_stack(np.where(endpoints))

    def _are_points_connected(
        self,
        skeleton: np.ndarray,
        point1: np.ndarray,
        point2: np.ndarray,
    ) -> bool:
        """Check if two points are connected in the skeleton.

        Args:
            skeleton: Binary skeleton image.
            point1: First point (y, x).
            point2: Second point (y, x).

        Returns:
            True if points are connected.
        """
        # Use connected components to check connectivity
        # Dilate slightly to handle near-connections
        dilated = binary_dilation(skeleton, iterations=1)
        _num_labels, labeled = cv2.connectedComponents(dilated.astype(np.uint8))

        label1 = labeled[point1[0], point1[1]]
        label2 = labeled[point2[0], point2[1]]

        return label1 == label2 and label1 > 0

    def _connect_skeleton_gaps(self, skeleton: np.ndarray) -> np.ndarray:
        """Connect nearby disconnected segments in a skeleton.

        Finds endpoints of skeleton segments and connects pairs that are
        within max_gap_distance of each other using linear interpolation.

        Args:
            skeleton: Binary skeleton image.

        Returns:
            Connected skeleton image.
        """
        skeleton = skeleton.copy()
        endpoints = self._find_skeleton_endpoints(skeleton)

        if len(endpoints) < 2:
            return skeleton

        # Find pairs of endpoints that should be connected
        connected: set[int] = set()

        for i, ep1 in enumerate(endpoints):
            if i in connected:
                continue

            best_j = None
            best_dist = float("inf")

            for j, ep2 in enumerate(endpoints):
                if j <= i or j in connected:
                    continue

                # Calculate distance
                dist = np.sqrt((ep1[0] - ep2[0]) ** 2 + (ep1[1] - ep2[1]) ** 2)

                if (
                    dist < best_dist
                    and dist <= self.max_gap_distance
                    and not self._are_points_connected(skeleton, ep1, ep2)
                ):
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                # Draw line between endpoints
                ep2 = endpoints[best_j]
                cv2.line(
                    skeleton,
                    (ep1[1], ep1[0]),  # cv2 uses (x, y)
                    (ep2[1], ep2[0]),
                    1,
                    thickness=1,
                )
                connected.add(i)
                connected.add(best_j)

        return skeleton

    def _order_points_nearest_neighbor(self, points: np.ndarray) -> np.ndarray:
        """Order points along a line using nearest neighbor traversal.

        Args:
            points: Unordered points as (N, 2) array of (x, y).

        Returns:
            Ordered points as (N, 2) array.
        """
        if len(points) <= 2:
            return points

        # Start from point with minimum x (or y if tied)
        start_idx = np.lexsort((points[:, 1], points[:, 0]))[0]

        ordered = [points[start_idx]]
        remaining = set(range(len(points)))
        remaining.remove(start_idx)

        current = points[start_idx]

        while remaining:
            # Find nearest unvisited point
            remaining_list = list(remaining)
            remaining_points = points[remaining_list]
            distances = np.sum((remaining_points - current) ** 2, axis=1)
            nearest_local_idx = np.argmin(distances)
            nearest_global_idx = remaining_list[nearest_local_idx]

            ordered.append(points[nearest_global_idx])
            remaining.remove(nearest_global_idx)
            current = points[nearest_global_idx]

        return np.array(ordered)

    def _extract_contour_from_skeleton(self, mask: np.ndarray) -> list[np.ndarray]:
        """Extract contours from a mask using skeletonization.

        Args:
            mask: Binary mask (H, W) with values 0 or 255.

        Returns:
            List of contours from this mask, each as (N, 1, 2) array.
        """
        skeleton = self._extract_skeleton(mask)

        # Find all skeleton points
        points = np.column_stack(np.where(skeleton > 0))  # (y, x) format

        if len(points) < self.min_length:
            return []

        # Convert to (x, y) format
        points_xy = points[:, ::-1]

        # Order points along the line
        ordered = self._order_points_nearest_neighbor(points_xy)

        # Apply epsilon approximation to reduce points
        contour = ordered.reshape(-1, 1, 2).astype(np.float32)
        length = cv2.arcLength(contour, closed=False)
        if length < self.min_length:
            return []

        epsilon = self.epsilon_factor * length
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)

        # Ensure shape is (N, 1, 2)
        if approx.ndim == 2:
            approx = approx.reshape(-1, 1, 2)

        return [approx.astype(np.int32)]

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
            if self.use_skeletonization:
                contours = self._extract_contour_from_skeleton(mask)
            else:
                contours = self._extract_contour_from_mask(mask)
            all_contours.extend(contours)

        return all_contours
