"""COCO instance segmentation format exporter.

Converts tile-based instance masks to COCO format for training
instance segmentation models like Mask2Former.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def mask_to_rle(mask: NDArray[np.uint8]) -> dict:
    """Convert binary mask to COCO RLE format.

    Uses pycocotools for efficient RLE encoding.

    Args:
        mask: Binary mask of shape (H, W) with values 0 or 1.

    Returns:
        RLE dict with 'counts' (string) and 'size' [H, W].
    """
    from pycocotools import mask as mask_utils

    # Ensure mask is binary and Fortran-ordered (required by pycocotools)
    binary_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)

    # Convert bytes to string for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_to_mask(rle: dict) -> NDArray[np.uint8]:
    """Convert COCO RLE format back to binary mask.

    Args:
        rle: RLE dict with 'counts' (string or bytes) and 'size' [H, W].

    Returns:
        Binary mask of shape (H, W) with values 0 or 1.
    """
    from pycocotools import mask as mask_utils

    # Ensure counts is bytes
    rle_copy = rle.copy()
    if isinstance(rle_copy["counts"], str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")

    return mask_utils.decode(rle_copy)


def mask_to_bbox(mask: NDArray[np.uint8]) -> list[float]:
    """Calculate bounding box from binary mask.

    Args:
        mask: Binary mask of shape (H, W).

    Returns:
        Bounding box as [x, y, width, height].
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return [0.0, 0.0, 0.0, 0.0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]


class COCOInstanceExporter:
    """Export instance masks to COCO format.

    Creates a COCO-format JSON file with images, annotations, and categories
    for instance segmentation training.

    Example:
        >>> exporter = COCOInstanceExporter(output_path=Path("coco.json"))
        >>> exporter.add_tile(Path("tile_0000_labels.json"))
        >>> exporter.add_tile(Path("tile_0001_labels.json"))
        >>> exporter.save()
    """

    def __init__(self, output_path: Path):
        """Initialize the exporter.

        Args:
            output_path: Path where the COCO JSON will be saved.
        """
        self.output_path = output_path

        # Initialize COCO structure
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._categories = [
            {
                "id": 1,
                "name": "contour_line",
                "supercategory": "contour",
            }
        ]

        self._next_image_id = 0
        self._next_annotation_id = 0

    def add_tile(self, labels_path: Path) -> None:
        """Add a tile to the COCO dataset.

        Reads the labels JSON file and corresponding instance mask,
        then creates COCO image and annotation entries.

        Args:
            labels_path: Path to the tile's labels JSON file.
        """
        with open(labels_path) as f:
            labels_data = json.load(f)

        # Get tile directory
        tile_dir = labels_path.parent

        # Extract image info
        image_info = labels_data.get("image", {})
        image_filename = image_info.get("path", "")
        size = image_info.get("size", {})
        height = size.get("height", 0)
        width = size.get("width", 0)

        if not image_filename or not height or not width:
            return

        # Add image entry
        image_id = self._next_image_id
        self._next_image_id += 1

        self._images.append(
            {
                "id": image_id,
                "file_name": image_filename,
                "height": height,
                "width": width,
            }
        )

        # Check for instance mask
        instance_mask_info = labels_data.get("instance_mask")
        if not instance_mask_info:
            return

        instance_mask_path = tile_dir / instance_mask_info.get("path", "")
        instances = instance_mask_info.get("instances", [])

        if not instance_mask_path.exists():
            return

        # Load instance mask
        instance_mask = cv2.imread(str(instance_mask_path), cv2.IMREAD_GRAYSCALE)
        if instance_mask is None:
            return

        # Create annotation for each instance
        for instance_info in instances:
            instance_id = instance_info.get("instance_id")
            elevation = instance_info.get("elevation")

            if instance_id is None:
                continue

            # Extract binary mask for this instance
            binary_mask = (instance_mask == instance_id).astype(np.uint8)

            if not np.any(binary_mask):
                continue

            # Calculate area and bbox
            area = int(np.sum(binary_mask))
            bbox = mask_to_bbox(binary_mask)

            # Convert to RLE
            rle = mask_to_rle(binary_mask)

            # Create annotation
            annotation = {
                "id": self._next_annotation_id,
                "image_id": image_id,
                "category_id": 1,  # contour_line
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }

            # Add elevation as custom field if available
            if elevation is not None:
                annotation["elevation"] = elevation

            self._annotations.append(annotation)
            self._next_annotation_id += 1

    def to_dict(self) -> dict:
        """Convert to COCO format dictionary.

        Returns:
            COCO-format dictionary with images, annotations, and categories.
        """
        return {
            "images": self._images,
            "annotations": self._annotations,
            "categories": self._categories,
        }

    def save(self) -> None:
        """Save the COCO dataset to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def convert_tiles_to_coco(
    tiles_dir: Path,
    output_path: Path,
    pattern: str = "*_labels.json",
) -> None:
    """Convert a directory of tiles to COCO format.

    Convenience function that finds all label files in a directory
    and exports them to a single COCO JSON file.

    Args:
        tiles_dir: Directory containing tile images and label files.
        output_path: Path for the output COCO JSON file.
        pattern: Glob pattern for finding label files.
    """
    exporter = COCOInstanceExporter(output_path=output_path)

    # Find all label files
    label_files = sorted(tiles_dir.glob(pattern))

    for label_file in label_files:
        exporter.add_tile(label_file)

    exporter.save()
