"""Training modules for contour segmentation models.

This package contains training implementations for:
- UNet: Semantic segmentation (binary mask prediction)
- Mask2Former: Instance segmentation (individual contour detection)
"""

from __future__ import annotations


# Lazy imports to avoid circular dependencies and allow partial imports
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "ContourDataset":
        from src.training.unet.dataset import ContourDataset

        return ContourDataset
    elif name == "TrainingConfig":
        from src.training.unet.config import TrainingConfig

        return TrainingConfig
    elif name == "unet":
        from src.training import unet

        return unet
    elif name == "mask2former":
        from src.training import mask2former

        return mask2former
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContourDataset",
    "TrainingConfig",
    "mask2former",
    "unet",
]
