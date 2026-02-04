"""Inference script for Mask2Former contour instance segmentation."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
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
