"""Package for generating training data for contour segmentation.

This package contains modules for generating synthetic contour maps with
text annotations and segmentation masks, including:
- Binary masks for UNet semantic segmentation
- Instance masks for Mask2Former instance segmentation
"""

from data_generation.coco_exporter import (
    COCOInstanceExporter,
    convert_tiles_to_coco,
    mask_to_rle,
    rle_to_mask,
)
from data_generation.tile_generator import (
    InstanceMaskResult,
    process_file,
    render_instance_mask,
    render_mask,
)

__all__ = [
    "COCOInstanceExporter",
    "InstanceMaskResult",
    "convert_tiles_to_coco",
    "mask_to_rle",
    "process_file",
    "render_instance_mask",
    "render_mask",
    "rle_to_mask",
]
