"""Tests for Mask2Former instance segmentation dataset."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from training.mask2former.config import Mask2FormerDatasetConfig
from training.mask2former.dataset import (
    Mask2FormerDataset,
    create_mask2former_dataloaders,
)


def create_sample_tile(
    tile_dir: Path, tile_id: str, num_instances: int = 3, size: int = 256
):
    """Create a sample tile with image, instance mask, and labels JSON."""
    # Create random image
    image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    image_path = tile_dir / f"{tile_id}.png"
    cv2.imwrite(str(image_path), image)

    # Create instance mask with num_instances horizontal lines
    instance_mask = np.zeros((size, size), dtype=np.uint8)
    instances = []
    for i in range(num_instances):
        instance_id = i + 1
        y_start = 30 + i * 60
        y_end = y_start + 4
        instance_mask[y_start:y_end, 30 : size - 30] = instance_id
        instances.append(
            {
                "instance_id": instance_id,
                "elevation": 100 + i * 10,
            }
        )

    instance_mask_path = tile_dir / f"{tile_id}_instance_mask.png"
    cv2.imwrite(str(instance_mask_path), instance_mask)

    # Create labels JSON
    labels_data = {
        "image": {"path": f"{tile_id}.png", "size": {"height": size, "width": size}},
        "instance_mask": {
            "path": f"{tile_id}_instance_mask.png",
            "instances": instances,
        },
        "labels": [],
    }
    labels_path = tile_dir / f"{tile_id}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f)

    return image_path, instance_mask_path, labels_path


class TestMask2FormerDatasetConfig:
    """Tests for the dataset configuration."""

    def test_config_default_values(self):
        """Config should have sensible defaults."""
        config = Mask2FormerDatasetConfig()

        assert config.image_size == 512
        assert config.num_labels == 1  # Just contour_line

    def test_config_custom_values(self):
        """Config should accept custom values."""
        config = Mask2FormerDatasetConfig(
            image_size=256,
            num_labels=2,
        )

        assert config.image_size == 256
        assert config.num_labels == 2


class TestMask2FormerDataset:
    """Tests for the Mask2Former dataset."""

    @pytest.fixture
    def temp_tiles_dir(self, tmp_path):
        """Create a temporary directory with sample tiles."""
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create 5 sample tiles
        for i in range(5):
            create_sample_tile(tiles_dir, f"tile_{i:04d}", num_instances=3)

        return tiles_dir

    def test_dataset_initialization(self, temp_tiles_dir):
        """Dataset should initialize and find all tiles."""
        dataset = Mask2FormerDataset(data_dir=temp_tiles_dir)

        assert len(dataset) == 5

    def test_dataset_getitem_returns_dict(self, temp_tiles_dir):
        """Dataset __getitem__ should return a dictionary."""
        dataset = Mask2FormerDataset(data_dir=temp_tiles_dir)

        item = dataset[0]

        assert isinstance(item, dict)
        assert "pixel_values" in item
        assert "pixel_mask" in item
        assert "mask_labels" in item
        assert "class_labels" in item

    def test_dataset_pixel_values_shape(self, temp_tiles_dir):
        """Pixel values should have correct shape [C, H, W]."""
        dataset = Mask2FormerDataset(data_dir=temp_tiles_dir, image_size=256)

        item = dataset[0]

        assert item["pixel_values"].shape == (3, 256, 256)
        assert item["pixel_values"].dtype == torch.float32

    def test_dataset_mask_labels_shape(self, temp_tiles_dir):
        """Mask labels should have shape [N, H, W] where N is num instances."""
        dataset = Mask2FormerDataset(data_dir=temp_tiles_dir, image_size=256)

        item = dataset[0]

        # Should have 3 instances
        assert item["mask_labels"].shape[0] == 3
        assert item["mask_labels"].shape[1] == 256
        assert item["mask_labels"].shape[2] == 256
        assert item["mask_labels"].dtype == torch.float32

    def test_dataset_class_labels(self, temp_tiles_dir):
        """Class labels should be tensor with one label per instance."""
        dataset = Mask2FormerDataset(data_dir=temp_tiles_dir, image_size=256)

        item = dataset[0]

        # Should have 3 class labels (all 0 for contour_line)
        assert item["class_labels"].shape == (3,)
        assert torch.all(item["class_labels"] == 0)  # All contour_line class

    def test_dataset_pixel_mask(self, temp_tiles_dir):
        """Pixel mask should indicate valid pixels."""
        dataset = Mask2FormerDataset(data_dir=temp_tiles_dir, image_size=256)

        item = dataset[0]

        assert item["pixel_mask"].shape == (256, 256)
        assert item["pixel_mask"].dtype == torch.long

    def test_dataset_handles_empty_instances(self, tmp_path):
        """Dataset should handle tiles with no instances."""
        tiles_dir = tmp_path / "empty_tiles"
        tiles_dir.mkdir()

        # Create tile with no instances
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(tiles_dir / "tile_0000.png"), image)
        cv2.imwrite(
            str(tiles_dir / "tile_0000_instance_mask.png"),
            np.zeros((256, 256), dtype=np.uint8),
        )
        labels_data = {
            "image": {"path": "tile_0000.png", "size": {"height": 256, "width": 256}},
            "instance_mask": {
                "path": "tile_0000_instance_mask.png",
                "instances": [],
            },
            "labels": [],
        }
        with open(tiles_dir / "tile_0000_labels.json", "w") as f:
            json.dump(labels_data, f)

        dataset = Mask2FormerDataset(data_dir=tiles_dir)

        item = dataset[0]

        # Should have 0 instances
        assert item["mask_labels"].shape[0] == 0
        assert item["class_labels"].shape[0] == 0


class TestCreateMask2FormerDataloaders:
    """Tests for dataloader creation."""

    @pytest.fixture
    def temp_tiles_dir(self, tmp_path):
        """Create a temporary directory with sample tiles."""
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        for i in range(10):
            create_sample_tile(tiles_dir, f"tile_{i:04d}", num_instances=2)

        return tiles_dir

    def test_create_dataloaders_returns_train_and_val(self, temp_tiles_dir):
        """Should return both train and validation dataloaders."""
        train_loader, val_loader = create_mask2former_dataloaders(
            data_dir=temp_tiles_dir,
            batch_size=2,
            val_split=0.2,
        )

        assert train_loader is not None
        assert val_loader is not None

    def test_dataloaders_batch_size(self, temp_tiles_dir):
        """Dataloaders should use specified batch size."""
        train_loader, val_loader = create_mask2former_dataloaders(
            data_dir=temp_tiles_dir,
            batch_size=2,
            val_split=0.2,
        )

        # Note: batch iteration would require collate_fn, just check creation
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2

    def test_dataloaders_split_sizes(self, temp_tiles_dir):
        """Train/val split should produce correct sizes."""
        train_loader, val_loader = create_mask2former_dataloaders(
            data_dir=temp_tiles_dir,
            batch_size=1,
            val_split=0.2,
        )

        # 10 tiles, 20% val = 2 val, 8 train
        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 2
