"""Common utilities shared across training modules."""

from training.common.augmentations import (
    get_training_augmentations,
    get_validation_augmentations,
)

# Aliases for compatibility
get_train_transforms = get_training_augmentations
get_val_transforms = get_validation_augmentations

__all__ = [
    "get_train_transforms",
    "get_training_augmentations",
    "get_val_transforms",
    "get_validation_augmentations",
]
