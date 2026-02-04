"""Mask2Former dataset configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mask2FormerDatasetConfig:
    """Configuration for Mask2Former instance segmentation dataset.

    Attributes:
        data_dir: Root directory containing tile subdirectories.
        image_size: Target image size (square).
        num_labels: Number of semantic classes (1 for contour_line only).
        model_name: Pretrained model name for processor initialization.
        val_split: Fraction of data to use for validation.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
    """

    data_dir: Path | None = None
    image_size: int = 512
    num_labels: int = 1  # Just contour_line class
    model_name: str = "facebook/mask2former-swin-tiny-coco-instance"
    val_split: float = 0.1
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class Mask2FormerTrainingConfig:
    """Configuration for Mask2Former training.

    Attributes:
        output_dir: Directory to save model checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        logging_steps: Log metrics every N steps.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        fp16: Use mixed precision training.
        backbone: Backbone model (e.g., 'swin-t', 'resnet50').
    """

    output_dir: Path = field(default_factory=lambda: Path("models/mask2former"))
    num_epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    backbone: str = "swin-t"
