"""Tests for ContourDataset."""

import cv2
import numpy as np
import pytest
import torch

from training.dataset import ContourDataset, create_train_val_split


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create a temporary directory with valid image-mask pairs."""
    # Create image and mask pairs
    for i in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (10, 50), (90, 50), 255, 3)

        cv2.imwrite(str(tmp_path / f"tile_{i}.png"), img)
        cv2.imwrite(str(tmp_path / f"tile_{i}_mask.png"), mask)

    return tmp_path


@pytest.fixture
def temp_nested_dataset_dir(tmp_path):
    """Create a temporary directory with nested subdirectories."""
    subdir1 = tmp_path / "region_a"
    subdir2 = tmp_path / "region_b"
    subdir1.mkdir()
    subdir2.mkdir()

    for i, subdir in enumerate([subdir1, subdir2]):
        for j in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mask = np.zeros((100, 100), dtype=np.uint8)

            cv2.imwrite(str(subdir / f"tile_{i}_{j}.png"), img)
            cv2.imwrite(str(subdir / f"tile_{i}_{j}_mask.png"), mask)

    return tmp_path


# --- Initialization Tests ---


def test_init_with_valid_directory(temp_dataset_dir):
    """Test initialization with valid data directory."""
    dataset = ContourDataset(temp_dataset_dir, image_size=256, is_train=True)

    assert len(dataset) == 5
    assert dataset.image_size == 256
    assert dataset.is_train is True


def test_init_with_nested_directories(temp_nested_dataset_dir):
    """Test initialization finds images in nested directories."""
    dataset = ContourDataset(temp_nested_dataset_dir, image_size=256)

    # Should find all 6 images (3 in each subdirectory)
    assert len(dataset) == 6


def test_init_with_empty_directory(tmp_path):
    """Test error when directory has no image-mask pairs."""
    with pytest.raises(ValueError, match="No image-mask pairs found"):
        ContourDataset(tmp_path)


def test_init_with_custom_image_paths(temp_dataset_dir):
    """Test initialization with explicit image paths."""
    # Only use first 2 images
    custom_paths = [
        temp_dataset_dir / "tile_0.png",
        temp_dataset_dir / "tile_1.png",
    ]

    dataset = ContourDataset(temp_dataset_dir, image_size=256, image_paths=custom_paths)

    assert len(dataset) == 2


def test_init_train_vs_validation_augmentations(temp_dataset_dir):
    """Test that train and validation use different augmentations."""
    train_dataset = ContourDataset(temp_dataset_dir, is_train=True)
    val_dataset = ContourDataset(temp_dataset_dir, is_train=False)

    # Transforms should be different objects
    assert train_dataset.transform is not val_dataset.transform


# --- Image-Mask Discovery Tests ---


def test_find_image_mask_pairs_skips_masks(temp_dataset_dir):
    """Test that mask files are not included as images."""
    dataset = ContourDataset(temp_dataset_dir)

    for path in dataset.image_paths:
        assert "_mask" not in path.stem


def test_find_image_mask_pairs_skips_debug(temp_dataset_dir):
    """Test that debug files are skipped."""
    # Create a debug image
    debug_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(temp_dataset_dir / "tile_debug.png"), debug_img)
    cv2.imwrite(str(temp_dataset_dir / "tile_debug_mask.png"), debug_img[:, :, 0])

    dataset = ContourDataset(temp_dataset_dir)

    for path in dataset.image_paths:
        assert "_debug" not in path.stem


def test_find_image_mask_pairs_requires_mask(temp_dataset_dir):
    """Test that images without masks are not included."""
    # Create image without mask
    orphan_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(temp_dataset_dir / "orphan.png"), orphan_img)

    dataset = ContourDataset(temp_dataset_dir)

    # Should still have only the original 5 pairs
    assert len(dataset) == 5


def test_image_paths_are_sorted(temp_dataset_dir):
    """Test that image paths are sorted for reproducibility."""
    dataset = ContourDataset(temp_dataset_dir)

    paths = [str(p) for p in dataset.image_paths]
    assert paths == sorted(paths)


# --- __len__ Tests ---


def test_len_returns_correct_count(temp_dataset_dir):
    """Test __len__ returns number of samples."""
    dataset = ContourDataset(temp_dataset_dir)

    assert len(dataset) == 5


def test_len_with_custom_paths(temp_dataset_dir):
    """Test __len__ with custom image paths."""
    custom_paths = [temp_dataset_dir / "tile_0.png"]
    dataset = ContourDataset(temp_dataset_dir, image_paths=custom_paths)

    assert len(dataset) == 1


# --- __getitem__ Tests ---


def test_getitem_returns_dict(temp_dataset_dir):
    """Test __getitem__ returns dictionary with image and mask."""
    dataset = ContourDataset(temp_dataset_dir, image_size=256)

    sample = dataset[0]

    assert isinstance(sample, dict)
    assert "image" in sample
    assert "mask" in sample


def test_getitem_image_is_tensor(temp_dataset_dir):
    """Test that returned image is a tensor."""
    dataset = ContourDataset(temp_dataset_dir, image_size=256)

    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)


def test_getitem_mask_is_tensor(temp_dataset_dir):
    """Test that returned mask is a tensor."""
    dataset = ContourDataset(temp_dataset_dir, image_size=256)

    sample = dataset[0]

    assert isinstance(sample["mask"], torch.Tensor)


def test_getitem_image_shape(temp_dataset_dir):
    """Test image tensor has correct shape [C, H, W]."""
    image_size = 256
    dataset = ContourDataset(temp_dataset_dir, image_size=image_size)

    sample = dataset[0]

    assert sample["image"].shape == (3, image_size, image_size)


def test_getitem_mask_shape(temp_dataset_dir):
    """Test mask tensor has correct shape [1, H, W]."""
    image_size = 256
    dataset = ContourDataset(temp_dataset_dir, image_size=image_size)

    sample = dataset[0]

    assert sample["mask"].shape == (1, image_size, image_size)


def test_getitem_mask_is_binary(temp_dataset_dir):
    """Test that mask values are binary (0 or 1)."""
    dataset = ContourDataset(temp_dataset_dir, image_size=256)

    sample = dataset[0]
    mask = sample["mask"]

    unique_values = torch.unique(mask)
    assert all(v in [0.0, 1.0] for v in unique_values.tolist())


def test_getitem_all_indices(temp_dataset_dir):
    """Test accessing all indices works."""
    dataset = ContourDataset(temp_dataset_dir)

    for i in range(len(dataset)):
        sample = dataset[i]
        assert "image" in sample
        assert "mask" in sample


def test_getitem_index_out_of_bounds(temp_dataset_dir):
    """Test accessing invalid index raises error."""
    dataset = ContourDataset(temp_dataset_dir)

    with pytest.raises(IndexError):
        _ = dataset[100]


# --- Error Handling Tests ---


def test_getitem_corrupted_image(temp_dataset_dir):
    """Test error when image file is corrupted."""
    # Create a corrupted image file
    corrupted_path = temp_dataset_dir / "corrupted.png"
    corrupted_path.write_bytes(b"not an image")
    mask_path = temp_dataset_dir / "corrupted_mask.png"
    cv2.imwrite(str(mask_path), np.zeros((100, 100), dtype=np.uint8))

    # Create dataset with only the corrupted image
    dataset = ContourDataset(temp_dataset_dir, image_paths=[corrupted_path])

    with pytest.raises(LookupError, match="Failed to load image"):
        _ = dataset[0]


def test_getitem_corrupted_mask(temp_dataset_dir):
    """Test error when mask file is corrupted."""
    # Create valid image but corrupted mask
    valid_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(temp_dataset_dir / "valid_img.png"), valid_img)

    corrupted_mask = temp_dataset_dir / "valid_img_mask.png"
    corrupted_mask.write_bytes(b"not a mask")

    dataset = ContourDataset(
        temp_dataset_dir, image_paths=[temp_dataset_dir / "valid_img.png"]
    )

    with pytest.raises(LookupError, match="Failed to load mask"):
        _ = dataset[0]


# --- get_image_path Tests ---


def test_get_image_path(temp_dataset_dir):
    """Test get_image_path returns correct path."""
    dataset = ContourDataset(temp_dataset_dir)

    path = dataset.get_image_path(0)

    assert path == dataset.image_paths[0]
    assert path.exists()


# --- Mask Binarization Tests ---


def test_mask_binarization_threshold(temp_dataset_dir):
    """Test that mask is binarized at threshold 127."""
    # Create mask with gradient values
    gradient_mask = np.arange(256, dtype=np.uint8).reshape(16, 16)
    gradient_mask = cv2.resize(gradient_mask, (100, 100))
    cv2.imwrite(str(temp_dataset_dir / "gradient_mask.png"), gradient_mask)

    # Create corresponding image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(temp_dataset_dir / "gradient.png"), img)

    dataset = ContourDataset(
        temp_dataset_dir,
        image_size=100,
        image_paths=[temp_dataset_dir / "gradient.png"],
    )

    sample = dataset[0]
    mask = sample["mask"]

    # All values should be 0 or 1
    unique = torch.unique(mask)
    assert len(unique) <= 2
    assert all(v in [0.0, 1.0] for v in unique.tolist())


# --- Train/Val Split Tests ---


def test_create_train_val_split(temp_dataset_dir):
    """Test train/val split creates two datasets."""
    train_ds, val_ds = create_train_val_split(temp_dataset_dir, val_split=0.2)

    assert isinstance(train_ds, ContourDataset)
    assert isinstance(val_ds, ContourDataset)


def test_create_train_val_split_sizes(temp_dataset_dir):
    """Test train/val split has correct sizes."""
    train_ds, val_ds = create_train_val_split(temp_dataset_dir, val_split=0.4, seed=42)

    total = len(train_ds) + len(val_ds)
    assert total == 5  # Original 5 images

    # Val should be ~40% = 2, train ~60% = 3
    assert len(val_ds) == 2
    assert len(train_ds) == 3


def test_create_train_val_split_no_overlap(temp_dataset_dir):
    """Test train and val sets have no overlapping images."""
    train_ds, val_ds = create_train_val_split(temp_dataset_dir, seed=42)

    train_paths = set(str(p) for p in train_ds.image_paths)
    val_paths = set(str(p) for p in val_ds.image_paths)

    assert len(train_paths & val_paths) == 0


def test_create_train_val_split_reproducible(temp_dataset_dir):
    """Test that same seed produces same split."""
    train1, val1 = create_train_val_split(temp_dataset_dir, seed=123)
    train2, val2 = create_train_val_split(temp_dataset_dir, seed=123)

    assert [str(p) for p in train1.image_paths] == [str(p) for p in train2.image_paths]
    assert [str(p) for p in val1.image_paths] == [str(p) for p in val2.image_paths]


def test_create_train_val_split_different_seeds(temp_dataset_dir):
    """Test that different seeds produce different splits."""
    train1, _ = create_train_val_split(temp_dataset_dir, seed=1)
    train2, _ = create_train_val_split(temp_dataset_dir, seed=2)

    # With 5 images and different seeds, splits should likely differ
    paths1 = [str(p) for p in train1.image_paths]
    paths2 = [str(p) for p in train2.image_paths]

    # Note: Small chance they could be same, but very unlikely
    assert paths1 != paths2 or len(train1) == 5  # Unless no split


def test_create_train_val_split_train_has_augmentations(temp_dataset_dir):
    """Test train dataset has training augmentations."""
    train_ds, val_ds = create_train_val_split(temp_dataset_dir)

    assert train_ds.is_train is True
    assert val_ds.is_train is False
