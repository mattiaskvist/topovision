"""Mask2Former training module for instance segmentation."""

from __future__ import annotations


# Lazy imports to avoid issues with different PYTHONPATH configurations
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "Mask2FormerDataset":
        from src.training.mask2former.dataset import Mask2FormerDataset

        return Mask2FormerDataset
    elif name == "Mask2FormerTrainingConfig":
        from src.training.mask2former.config import Mask2FormerTrainingConfig

        return Mask2FormerTrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Mask2FormerDataset", "Mask2FormerTrainingConfig"]
