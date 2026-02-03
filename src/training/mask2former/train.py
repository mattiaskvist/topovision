"""Training script for Mask2Former contour instance segmentation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from transformers import (
        Mask2FormerForUniversalSegmentation,
    )
except ImportError as e:
    msg = "transformers not installed. Run: pip install transformers>=4.35.0"
    raise ImportError(msg) from e

from training.mask2former.config import Mask2FormerTrainingConfig
from training.mask2former.dataset import (
    create_mask2former_dataloaders,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def create_model(
    num_labels: int = 1,
    pretrained: str = "facebook/mask2former-swin-tiny-coco-instance",
) -> Mask2FormerForUniversalSegmentation:
    """Create Mask2Former model for instance segmentation.

    Args:
        num_labels: Number of semantic classes.
        pretrained: Pretrained model name or path.

    Returns:
        Configured Mask2Former model.
    """
    # Load pretrained model and modify for our task
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


def compute_metrics(
    model: Mask2FormerForUniversalSegmentation,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute validation metrics.

    Args:
        model: The model to evaluate.
        dataloader: Validation dataloader.
        device: Device to use.

    Returns:
        Dictionary of metric names to values.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    return {
        "val_loss": total_loss / max(num_batches, 1),
    }


def train_epoch(
    model: Mask2FormerForUniversalSegmentation,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None = None,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to use.
        epoch: Current epoch number.
        writer: TensorBoard writer.
        gradient_accumulation_steps: Number of steps to accumulate gradients.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress):
        # Move to device
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        # Handle variable-length mask_labels and class_labels
        # Split batched tensors into list of per-sample tensors
        batch_size = pixel_values.shape[0]
        mask_labels = []
        class_labels = []

        for i in range(batch_size):
            # Get valid masks (class_labels != -1)
            valid = batch["class_labels"][i] != -1
            mask_labels.append(batch["mask_labels"][i][valid].to(device))
            class_labels.append(batch["class_labels"][i][valid].to(device))

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()
        num_batches += 1

        # Update progress bar
        progress.set_postfix(
            loss=f"{outputs.loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        # Log to TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("train/loss", outputs.loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    return total_loss / max(num_batches, 1)


def train(
    config: Mask2FormerTrainingConfig,
    data_dir: Path,
    val_split: float = 0.1,
    resume_from: Path | None = None,
) -> Path:
    """Run full training.

    Args:
        config: Training configuration.
        data_dir: Directory containing training data.
        val_split: Fraction of data for validation.
        resume_from: Optional checkpoint to resume from.

    Returns:
        Path to the best model checkpoint.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    run_name = f"mask2former_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = config.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print(f"Loading data from {data_dir}")
    train_loader, val_loader = create_mask2former_dataloaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        val_split=val_split,
        image_size=512,
        num_workers=4,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating Mask2Former model...")
    model = create_model(num_labels=1)  # Just contour_line
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from is not None and resume_from.exists():
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")

    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            writer=writer,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Validate
        val_metrics = compute_metrics(model, val_loader, device)
        val_loss = val_metrics["val_loss"]

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Log to TensorBoard
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")

            # Also save in HuggingFace format for easy loading
            model.save_pretrained(output_dir / "best_model_hf")

    writer.close()
    print(f"Training complete. Best model: {output_dir / 'best_model.pt'}")

    return output_dir / "best_model.pt"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Mask2Former for contour instance segmentation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing training data with instance masks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/mask2former"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    config = Mask2FormerTrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    train(
        config=config,
        data_dir=args.data_dir,
        val_split=args.val_split,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
