"""UNet training module for semantic segmentation."""

from __future__ import annotations


# Lazy imports to avoid issues with different PYTHONPATH configurations
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "ContourDataset":
        from src.training.unet.dataset import ContourDataset

        return ContourDataset
    elif name == "TrainingConfig":
        from src.training.unet.config import TrainingConfig

        return TrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ContourDataset", "TrainingConfig"]
