"""PyTorch Dataset for contour segmentation training."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.common.augmentations import (
    get_training_augmentations,
    get_validation_augmentations,
)


class ContourDataset(Dataset):
    """Dataset for contour line segmentation.

    Loads tile images and their corresponding binary masks for training
    a U-Net segmentation model.

    Args:
        data_dir: Root directory containing tile subdirectories.
        image_size: Target image size (square).
        is_train: If True, apply training augmentations; else validation only.
        image_paths: Optional list of specific image paths to use.
    """

    def __init__(
        self,
        data_dir: Path | str,
        image_size: int = 512,
        is_train: bool = True,
        image_paths: list[Path] | None = None,
    ):
        """Initialize the dataset."""
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_train = is_train

        # Set up augmentations
        if is_train:
            self.transform = get_training_augmentations(image_size)
        else:
            self.transform = get_validation_augmentations(image_size)

        # Find all image-mask pairs
        if image_paths is not None:
            self.image_paths = [Path(p) for p in image_paths]
        else:
            self.image_paths = self._find_image_mask_pairs()

    def _find_image_mask_pairs(self) -> list[Path]:
        """Find all valid image-mask pairs in the data directory.

        Returns:
            List of image paths that have corresponding masks.
        """
        image_paths = []

        # Search recursively for tile images (not masks, not debug)
        for img_path in self.data_dir.rglob("*.png"):
            # Skip masks and debug images
            if "_mask" in img_path.stem or "_debug" in img_path.stem:
                continue

            # Check if corresponding mask exists
            mask_path = img_path.parent / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                image_paths.append(img_path)

        if not image_paths:
            msg = f"No image-mask pairs found in {self.data_dir}"
            raise ValueError(msg)

        return sorted(image_paths)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'image' and 'mask' tensors.
        """
        img_path = self.image_paths[idx]
        mask_path = img_path.parent / f"{img_path.stem}_mask.png"

        # Load image (BGR -> RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to load image: {img_path}"
            raise LookupError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Failed to load mask: {mask_path}"
            raise LookupError(msg)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        # Ensure mask has channel dimension [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask}

    def get_image_path(self, idx: int) -> Path:
        """Get the path for a specific sample index."""
        return self.image_paths[idx]


def create_train_val_split(
    data_dir: Path | str,
    image_size: int = 512,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[ContourDataset, ContourDataset]:
    """Create train and validation datasets with a random split.

    Args:
        data_dir: Root directory containing tile subdirectories.
        image_size: Target image size.
        val_split: Fraction of data for validation (0-1).
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Create a temporary dataset to get all paths
    temp_dataset = ContourDataset(data_dir, image_size, is_train=True)
    all_paths = temp_dataset.image_paths

    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_paths))

    val_size = int(len(all_paths) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_paths = [all_paths[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]

    # Create datasets
    train_dataset = ContourDataset(
        data_dir, image_size, is_train=True, image_paths=train_paths
    )
    val_dataset = ContourDataset(
        data_dir, image_size, is_train=False, image_paths=val_paths
    )

    return train_dataset, val_dataset
