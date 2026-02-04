"""Inference script for Mask2Former contour instance segmentation."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from scipy.interpolate import splev, splprep
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


def load_model(
    model_path: str = "mattiaskvist/topovision-segmentation",
    subfolder: str = "mask2former",
    device: str | None = None,
) -> tuple[
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, torch.device
]:
    """Load Mask2Former model from HuggingFace Hub or local path.

    Args:
        model_path: HuggingFace Hub model ID or local path.
        subfolder: Subfolder within the model repository.
        device: Device to use (auto-detected if None).

    Returns:
        Tuple of (model, processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Loading model from {model_path}/{subfolder}...")
    print(f"Using device: {device}")

    # Load processor (use base Mask2Former processor)
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-tiny-coco-instance"
    )

    # Load model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path,
        subfolder=subfolder,
    )
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, processor, device


def predict(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run inference on a single image.

    Args:
        model: Loaded Mask2Former model.
        processor: Image processor.
        image: RGB image as numpy array (H, W, 3).
        device: Device to use.
        threshold: Confidence threshold for instance masks.

    Returns:
        Dictionary with:
        - masks: List of binary masks (H, W)
        - scores: Confidence scores
        - labels: Class labels
    """
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get instance masks
    # Target size should match input image
    target_size = (image.shape[0], image.shape[1])

    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[target_size],
        threshold=threshold,
    )[0]

    masks = []
    scores = []
    labels = []

    for segment in results["segments_info"]:
        mask = (results["segmentation"] == segment["id"]).cpu().numpy()
        masks.append(mask)
        scores.append(segment["score"])
        labels.append(segment["label_id"])

    return {
        "masks": masks,
        "scores": scores,
        "labels": labels,
    }


def extract_skeletons(
    masks: list[np.ndarray],
    connect_gaps: bool = True,
    max_gap_distance: int = 15,
) -> list[np.ndarray]:
    """Extract single-pixel wide skeletons from instance masks.

    Uses morphological skeletonization to reduce each contour mask
    to its medial axis (spine), with optional gap connection.

    Args:
        masks: List of binary masks (H, W) from predict().
        connect_gaps: Whether to connect nearby disconnected skeleton segments.
        max_gap_distance: Maximum pixel distance to bridge between segments.

    Returns:
        List of skeleton masks (H, W), each with single-pixel wide lines.
    """
    skeletons = []
    for mask in masks:
        # Skeletonize expects binary image
        skeleton = skeletonize(mask.astype(bool))
        skeleton = skeleton.astype(np.uint8)

        if connect_gaps:
            skeleton = _connect_skeleton_gaps(skeleton, max_gap_distance)

        skeletons.append(skeleton)
    return skeletons


def _find_skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
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


def _connect_skeleton_gaps(
    skeleton: np.ndarray,
    max_distance: int = 15,
) -> np.ndarray:
    """Connect nearby disconnected segments in a skeleton.

    Finds endpoints of skeleton segments and connects pairs that are
    within max_distance of each other using linear interpolation.

    Args:
        skeleton: Binary skeleton image.
        max_distance: Maximum pixel distance to bridge.

    Returns:
        Connected skeleton image.
    """
    skeleton = skeleton.copy()
    endpoints = _find_skeleton_endpoints(skeleton)

    if len(endpoints) < 2:
        return skeleton

    # Find pairs of endpoints that should be connected
    connected = set()

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
                and dist <= max_distance
                and not _are_connected(skeleton, ep1, ep2)
            ):
                # Check if these endpoints belong to different components
                # by checking if there's already a path between them

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


def _are_connected(
    skeleton: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray,
) -> bool:
    """Check if two points are connected in the skeleton via flood fill.

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


def _order_points_nearest_neighbor(points: np.ndarray) -> np.ndarray:
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
        remaining_points = points[list(remaining)]
        distances = np.sum((remaining_points - current) ** 2, axis=1)
        nearest_local_idx = np.argmin(distances)
        nearest_global_idx = list(remaining)[nearest_local_idx]

        ordered.append(points[nearest_global_idx])
        remaining.remove(nearest_global_idx)
        current = points[nearest_global_idx]

    return np.array(ordered)


def masks_to_polylines(
    skeletons: list[np.ndarray],
    min_length: int = 10,
    smooth: bool = False,
    smoothing_factor: float = 0.0,
    num_points: int | None = None,
) -> list[np.ndarray]:
    """Convert skeleton masks to ordered polyline coordinates.

    Args:
        skeletons: List of skeleton masks from extract_skeletons().
        min_length: Minimum number of points to keep a polyline.
        smooth: Whether to apply spline smoothing to polylines.
        smoothing_factor: Smoothing factor for spline (0 = interpolate exactly).
        num_points: Number of points in output polyline (None = same as input).

    Returns:
        List of polylines, each as array of (x, y) coordinates.
    """
    polylines = []

    for skeleton in skeletons:
        # Find all skeleton points
        points = np.column_stack(np.where(skeleton > 0))  # (y, x) format

        if len(points) < min_length:
            continue

        # Convert to (x, y) format
        points = points[:, ::-1]  # Swap to (x, y)

        # Order points along the line using nearest neighbor
        ordered = _order_points_nearest_neighbor(points)

        if smooth and len(ordered) >= 4:
            ordered = _smooth_polyline(ordered, smoothing_factor, num_points)

        polylines.append(ordered)

    return polylines


def _smooth_polyline(
    points: np.ndarray,
    smoothing_factor: float = 0.0,
    num_points: int | None = None,
) -> np.ndarray:
    """Smooth a polyline using B-spline interpolation.

    Args:
        points: Ordered points as (N, 2) array of (x, y).
        smoothing_factor: Smoothing factor (0 = exact interpolation).
        num_points: Number of points in output (None = same as input).

    Returns:
        Smoothed polyline as (M, 2) array.
    """
    if num_points is None:
        num_points = len(points)

    try:
        # Fit a B-spline to the points
        # k=3 for cubic spline, but reduce if not enough points
        k = min(3, len(points) - 1)
        tck, _u = splprep([points[:, 0], points[:, 1]], s=smoothing_factor, k=k)

        # Evaluate spline at evenly spaced points
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)

        return np.column_stack([x_new, y_new])
    except Exception:
        # Fall back to original if spline fitting fails
        return points


def predict_with_skeletons(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    connect_gaps: bool = True,
    max_gap_distance: int = 15,
    smooth_polylines: bool = False,
) -> dict:
    """Run inference and extract skeletons in one call.

    Args:
        model: Loaded Mask2Former model.
        processor: Image processor.
        image: RGB image as numpy array (H, W, 3).
        device: Device to use.
        threshold: Confidence threshold.
        connect_gaps: Whether to connect disconnected skeleton segments.
        max_gap_distance: Maximum pixel distance to bridge between segments.
        smooth_polylines: Whether to apply spline smoothing to polylines.

    Returns:
        Dictionary with masks, skeletons, polylines, scores, labels.
    """
    predictions = predict(model, processor, image, device, threshold)

    skeletons = extract_skeletons(
        predictions["masks"],
        connect_gaps=connect_gaps,
        max_gap_distance=max_gap_distance,
    )
    polylines = masks_to_polylines(skeletons, smooth=smooth_polylines)

    predictions["skeletons"] = skeletons
    predictions["polylines"] = polylines

    return predictions


def visualize_skeletons(
    image: np.ndarray,
    predictions: dict,
    line_thickness: int = 1,
) -> np.ndarray:
    """Visualize skeleton lines overlaid on image.

    Args:
        image: Original RGB image.
        predictions: Output from predict_with_skeletons().
        line_thickness: Thickness of skeleton lines.

    Returns:
        Visualization image (RGB).
    """
    vis = image.copy()

    # Generate random colors for each instance
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(predictions.get("skeletons", [])), 3))

    for i, skeleton in enumerate(predictions.get("skeletons", [])):
        color = colors[i].tolist()
        # Draw skeleton pixels
        vis[skeleton > 0] = color

        # Optionally thicken the line
        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            dilated = cv2.dilate(skeleton, kernel, iterations=1)
            vis[dilated > 0] = color

    return vis


def visualize_predictions(
    image: np.ndarray,
    predictions: dict,
    alpha: float = 0.5,
) -> np.ndarray:
    """Visualize predictions overlaid on image.

    Args:
        image: Original RGB image.
        predictions: Output from predict().
        alpha: Transparency for mask overlay.

    Returns:
        Visualization image (RGB).
    """
    vis = image.copy()

    # Generate random colors for each instance
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(predictions["masks"]), 3))

    for i, mask in enumerate(predictions["masks"]):
        color = colors[i]
        # Create colored mask
        colored_mask = np.zeros_like(vis)
        colored_mask[mask] = color

        # Blend with original image
        vis = np.where(
            mask[:, :, None],
            (1 - alpha) * vis + alpha * colored_mask,
            vis,
        ).astype(np.uint8)

        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color.tolist(), 2)

    return vis


def run_inference_on_file(
    image_path: str,
    model_path: str = "mattiaskvist/topovision-segmentation",
    subfolder: str = "mask2former",
    output_path: str | None = None,
    threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    """Run inference on an image file.

    Args:
        image_path: Path to input image.
        model_path: HuggingFace Hub model ID or local path.
        subfolder: Subfolder within the model repository.
        output_path: Path to save visualization (optional).
        threshold: Confidence threshold.
        device: Device to use.

    Returns:
        Predictions dictionary.
    """
    # Load model
    model, processor, device = load_model(model_path, subfolder, device)

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Running inference on {image_path}...")
    predictions = predict(model, processor, image, device, threshold)

    print(f"Found {len(predictions['masks'])} contour instances")
    for i, score in enumerate(predictions["scores"]):
        print(f"  Instance {i}: score={score:.3f}")

    # Save visualization
    if output_path:
        vis = visualize_predictions(image, predictions)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        print(f"Saved visualization to {output_path}")

    return predictions


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run Mask2Former inference for contour segmentation"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="mattiaskvist/topovision-segmentation",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="mask2former",
        help="Subfolder within model repository",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda, auto-detected if not specified)",
    )

    args = parser.parse_args()

    run_inference_on_file(
        image_path=args.image,
        model_path=args.model,
        subfolder=args.subfolder,
        output_path=args.output,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
