"""Training script for contour segmentation U-Net."""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    msg = (
        "segmentation-models-pytorch not installed. "
        "Run: pip install segmentation-models-pytorch"
    )
    raise ImportError(msg) from e

from .config import TrainingConfig
from .dataset import create_train_val_split
from .losses import DiceBCELoss


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        logits: Model output logits [B, 1, H, W].
        targets: Ground truth masks [B, 1, H, W].
        threshold: Probability threshold for binary prediction.

    Returns:
        Dictionary with IoU and Dice metrics.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # IoU (Intersection over Union)
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        # Dice (F1)
        dice = (2.0 * intersection + 1e-6) / (
            preds_flat.sum() + targets_flat.sum() + 1e-6
        )

        return {"iou": iou.item(), "dice": dice.item()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Returns:
        Dictionary with average loss and metrics.
    """
    model.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss, loss_components = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        # Metrics
        metrics = compute_metrics(logits, masks)

        total_loss += loss.item()
        total_dice_loss += loss_components["dice"]
        total_bce_loss += loss_components["bce"]
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "dice_loss": total_dice_loss / num_batches,
        "bce_loss": total_bce_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate the model.

    Returns:
        Dictionary with average loss and metrics.
    """
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss, _ = criterion(logits, masks)
        metrics = compute_metrics(logits, masks)

        total_loss += loss.item()
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        num_batches += 1

    # Avoid division by zero
    if num_batches == 0:
        print("Warning: No batches in validation loader.")
        return {
            "loss": 0.0,
            "iou": 0.0,
            "dice": 0.0,
        }

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


def train(config: TrainingConfig):
    """Run the full training loop.

    Args:
        config: Training configuration.
    """
    print("=" * 60)
    print("Contour Segmentation U-Net Training")
    print("=" * 60)

    # Setup device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Create datasets
    print(f"\nLoading data from: {config.data_dir}")
    train_dataset, val_dataset = create_train_val_split(
        config.data_dir,
        config.image_size,
        config.val_split,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
    print(f"\nCreating U-Net with {config.encoder_name} encoder")
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=1,
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = DiceBCELoss(config.dice_weight, config.bce_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    # TensorBoard
    writer = SummaryWriter(run_dir / "tensorboard")

    # Training loop
    best_val_dice = 0.0
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/dice", train_metrics["dice"], epoch)
        writer.add_scalar("train/iou", train_metrics["iou"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{config.num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Dice: {val_metrics['dice']:.4f} "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        # Save best model
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                    "config": {
                        "encoder_name": config.encoder_name,
                        "encoder_weights": config.encoder_weights,
                        "image_size": config.image_size,
                    },
                },
                run_dir / "best_model.pt",
            )
            print(f"  → Saved new best model (Dice: {best_val_dice:.4f})")

        # Save periodic checkpoint
        if epoch % config.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )

    # Save final model
    torch.save(
        {
            "epoch": config.num_epochs,
            "model_state_dict": model.state_dict(),
            "config": {
                "encoder_name": config.encoder_name,
                "encoder_weights": config.encoder_weights,
                "image_size": config.image_size,
            },
        },
        run_dir / "final_model.pt",
    )

    writer.close()

    print("-" * 60)
    print(f"Training complete! Best validation Dice: {best_val_dice:.4f}")
    print(f"Models saved to: {run_dir}")


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train contour segmentation U-Net")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dataVisualization/output"),
        help="Directory containing training tiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet-b0",
            "efficientnet-b2",
        ],
        help="Encoder backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Training device",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Validation split fraction"
    )

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        encoder_name=args.encoder,
        device=args.device,
        val_split=args.val_split,
    )

    train(config)


if __name__ == "__main__":
    main()
