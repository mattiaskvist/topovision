"""Training module for contour line segmentation models.

Supports:
- UNet semantic segmentation (training.unet)
- Mask2Former instance segmentation (training.mask2former)
"""

# Re-export commonly used items for backward compatibility
from training.unet import ContourDataset, TrainingConfig

__all__ = ["ContourDataset", "TrainingConfig"]
