"""Training configuration for contour segmentation U-Net."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for U-Net training.

    Attributes:
        data_dir: Directory containing training tiles and masks.
        output_dir: Directory for saving checkpoints and logs.
        image_size: Input image size (square).
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: AdamW weight decay.
        encoder_name: Backbone encoder for U-Net (e.g., "resnet34", "efficientnet-b0").
        encoder_weights: Pretrained weights (e.g., "imagenet" or None).
        val_split: Fraction of data for validation.
        num_workers: DataLoader workers.
        device: Training device ("cuda", "mps", or "cpu").
        save_every: Save checkpoint every N epochs.
        dice_weight: Weight for Dice loss component.
        bce_weight: Weight for BCE loss component.
    """

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data/training"))
    output_dir: Path = field(default_factory=lambda: Path("models"))

    # Image settings
    image_size: int = 512

    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Model architecture
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"

    # Data split
    val_split: float = 0.15

    # DataLoader
    num_workers: int = 4

    # Device
    device: str = "cuda"

    # Checkpointing
    save_every: int = 10

    # Loss weights
    dice_weight: float = 1.0
    bce_weight: float = 1.0

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
