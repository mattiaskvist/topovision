"""Test trained U-Net model on sample images."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch


def load_model(
    model_path: str, encoder: str = "resnet34", device: str = "cpu"
) -> torch.nn.Module:
    """Load trained U-Net model."""
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Dice: {checkpoint.get('val_dice', 'unknown'):.4f}")

    return model


def predict(
    model: torch.nn.Module, image: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Run inference on a single image."""
    # Preprocess: resize to 512x512, normalize to [0,1], convert to tensor
    original_size = image.shape[:2]
    resized = cv2.resize(image, (512, 512))

    # Normalize and convert to tensor (C, H, W)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Inference
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()

    # Resize back to original size
    prediction = cv2.resize(prediction, (original_size[1], original_size[0]))

    return prediction


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
    threshold: float = 0.5,
    save_path: str | None = None,
) -> None:
    """Visualize the prediction alongside the input image."""
    binary_pred = (prediction > threshold).astype(np.uint8) * 255

    if ground_truth is not None:
        _fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = [
            "Input Image",
            "Ground Truth",
            "Prediction (raw)",
            "Prediction (binary)",
        ]
        images = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            ground_truth,
            prediction,
            binary_pred,
        ]
    else:
        _fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Input Image", "Prediction (raw)", "Prediction (binary)"]
        images = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            prediction,
            binary_pred,
        ]

    for ax, img, title in zip(axes, images, titles, strict=True):
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main() -> None:
    """Main function to parse arguments and run model evaluation."""
    parser = argparse.ArgumentParser(description="Test trained U-Net model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g., models/run_xxx/best_model.pt)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (or directory of images)",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to ground truth mask (optional, for comparison)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/predictions",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Encoder architecture (must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary prediction",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model, args.encoder, args.device)

    # Process image(s)
    image_path = Path(args.image)

    if image_path.is_dir():
        # Process all images in directory
        image_files = list(image_path.glob("*_image.png"))
        print(f"Found {len(image_files)} images to process")

        for img_file in image_files[:5]:  # Process first 5
            image = cv2.imread(str(img_file))
            prediction = predict(model, image, args.device)

            # Look for corresponding mask
            mask_file = img_file.with_name(
                img_file.name.replace("_image.png", "_mask.png")
            )
            ground_truth = None
            if mask_file.exists():
                ground_truth = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

            save_path = output_dir / f"{img_file.stem}_prediction.png"
            visualize_prediction(
                image, prediction, ground_truth, args.threshold, str(save_path)
            )
    else:
        # Process single image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        prediction = predict(model, image, args.device)

        # Load ground truth if provided
        ground_truth = None
        if args.mask:
            ground_truth = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

        save_path = output_dir / f"{image_path.stem}_prediction.png"
        visualize_prediction(
            image, prediction, ground_truth, args.threshold, str(save_path)
        )


if __name__ == "__main__":
    main()
