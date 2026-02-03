"""PyTorch Dataset for Mask2Former instance segmentation training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

if TYPE_CHECKING:
    pass


class Mask2FormerDataset(Dataset):
    """Dataset for Mask2Former contour line instance segmentation.

    Loads tile images and their corresponding instance masks for training
    a Mask2Former instance segmentation model.

    The dataset expects tiles generated with `--instance-mask` flag, containing:
    - {tile_id}.png: RGB image
    - {tile_id}_instance_mask.png: Instance ID mask (uint8, 0=background)
    - {tile_id}_labels.json: Labels with instance info

    Args:
        data_dirs: Directory or list of directories containing tile files.
        image_size: Target image size for resizing.
        use_processor: If True, use Mask2FormerImageProcessor for preprocessing.
        processor: Optional pre-initialized processor.
    """

    def __init__(
        self,
        data_dirs: Path | str | list[Path | str],
        image_size: int = 512,
        use_processor: bool = False,
        processor=None,
    ):
        """Initialize the dataset.

        Args:
            data_dirs: Directory or list of directories containing tile files.
            image_size: Target image size for resizing.
            use_processor: If True, use Mask2FormerImageProcessor for preprocessing.
            processor: Optional pre-initialized processor.
        """
        # Handle single dir or list of dirs
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(d) for d in data_dirs]
        self.image_size = image_size
        self.use_processor = use_processor
        self.processor = processor

        # Find all label files from all directories
        self.label_files = []
        for data_dir in self.data_dirs:
            self.label_files.extend(sorted(data_dir.glob("*_labels.json")))

        # Filter to only include tiles with instance masks
        self.samples: list[dict] = []
        for label_file in self.label_files:
            with open(label_file) as f:
                data = json.load(f)

            if "instance_mask" not in data:
                continue

            # Get the directory containing this label file
            tile_dir = label_file.parent
            image_path = tile_dir / data["image"]["path"]
            instance_mask_path = tile_dir / data["instance_mask"]["path"]

            if image_path.exists() and instance_mask_path.exists():
                self.samples.append(
                    {
                        "image_path": image_path,
                        "instance_mask_path": instance_mask_path,
                        "instances": data["instance_mask"].get("instances", []),
                        "size": data["image"]["size"],
                    }
                )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with:
            - pixel_values: Image tensor [C, H, W]
            - pixel_mask: Valid pixel mask [H, W]
            - mask_labels: Instance masks [N, H, W]
            - class_labels: Class IDs for each instance [N]
        """
        sample = self.samples[idx]

        # Load image (BGR to RGB)
        image = cv2.imread(str(sample["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load instance mask
        instance_mask = cv2.imread(
            str(sample["instance_mask_path"]), cv2.IMREAD_GRAYSCALE
        )

        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            instance_mask = cv2.resize(
                instance_mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,  # Preserve instance IDs
            )

        # Get unique instance IDs (excluding background 0)
        instance_ids = [inst["instance_id"] for inst in sample["instances"]]

        # Create binary masks for each instance
        if instance_ids:
            masks = np.stack(
                [(instance_mask == iid).astype(np.float32) for iid in instance_ids],
                axis=0,
            )
            # All instances are contour_line class (class 0)
            class_labels = np.zeros(len(instance_ids), dtype=np.int64)
        else:
            # No instances - empty tensors
            masks = np.zeros((0, self.image_size, self.image_size), dtype=np.float32)
            class_labels = np.zeros(0, dtype=np.int64)

        # Normalize image
        pixel_values = image.astype(np.float32) / 255.0
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        pixel_values = (pixel_values - mean) / std
        # Convert to CHW format
        pixel_values = pixel_values.transpose(2, 0, 1)

        # Create pixel mask (all valid for now)
        pixel_mask = np.ones((self.image_size, self.image_size), dtype=np.int64)

        return {
            "pixel_values": torch.from_numpy(pixel_values).float(),
            "pixel_mask": torch.from_numpy(pixel_mask).long(),
            "mask_labels": torch.from_numpy(masks).float(),
            "class_labels": torch.from_numpy(class_labels).long(),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable number of instances.

    Handles batching of samples with different numbers of instances
    by padding mask_labels and class_labels.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with padded tensors.
    """
    # Stack fixed-size tensors
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])

    # Find max number of instances
    max_instances = max(item["mask_labels"].shape[0] for item in batch)

    if max_instances == 0:
        # No instances in batch
        batch_size = len(batch)
        h, w = batch[0]["pixel_mask"].shape
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "mask_labels": torch.zeros((batch_size, 0, h, w)),
            "class_labels": torch.zeros((batch_size, 0), dtype=torch.long),
        }

    # Pad mask_labels and class_labels
    h, w = batch[0]["pixel_mask"].shape
    padded_masks = []
    padded_labels = []

    for item in batch:
        n = item["mask_labels"].shape[0]
        if n < max_instances:
            # Pad masks with zeros
            pad_masks = torch.zeros((max_instances - n, h, w))
            padded_masks.append(torch.cat([item["mask_labels"], pad_masks], dim=0))
            # Pad labels with -1 (ignore index)
            pad_labels = torch.full((max_instances - n,), -1, dtype=torch.long)
            padded_labels.append(torch.cat([item["class_labels"], pad_labels], dim=0))
        else:
            padded_masks.append(item["mask_labels"])
            padded_labels.append(item["class_labels"])

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": torch.stack(padded_masks),
        "class_labels": torch.stack(padded_labels),
    }


def create_mask2former_dataloaders(
    data_dirs: Path | str | list[Path | str],
    batch_size: int = 4,
    val_split: float = 0.1,
    image_size: int = 512,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dirs: Directory or list of directories containing tile files.
        batch_size: Batch size for both dataloaders.
        val_split: Fraction of data for validation.
        image_size: Target image size.
        num_workers: Number of data loading workers.
        seed: Random seed for split reproducibility.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # Create full dataset
    dataset = Mask2FormerDataset(data_dirs=data_dirs, image_size=image_size)

    # Calculate split sizes
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
