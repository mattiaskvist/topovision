"""Tests for COCO instance segmentation exporter."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Import will work after we implement the module
try:
    from data_generation.coco_exporter import (
        COCOInstanceExporter,
        convert_tiles_to_coco,
        mask_to_rle,
        rle_to_mask,
    )
except ImportError:
    COCOInstanceExporter = None
    convert_tiles_to_coco = None
    mask_to_rle = None
    rle_to_mask = None


class TestMaskToRLE:
    """Tests for RLE encoding/decoding of masks."""

    @pytest.mark.skipif(mask_to_rle is None, reason="Module not implemented")
    def test_mask_to_rle_simple(self):
        """Simple binary mask should encode to RLE format."""
        # Create a simple 8x8 mask with a horizontal line
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3, 2:6] = 1  # Horizontal line

        rle = mask_to_rle(mask)

        assert "counts" in rle
        assert "size" in rle
        assert rle["size"] == [8, 8]

    @pytest.mark.skipif(mask_to_rle is None, reason="Module not implemented")
    def test_rle_roundtrip(self):
        """RLE encoding then decoding should return original mask."""
        # Create a more complex mask
        mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(mask, (16, 16), 8, 1, -1)

        rle = mask_to_rle(mask)
        decoded = rle_to_mask(rle)

        np.testing.assert_array_equal(mask, decoded)

    @pytest.mark.skipif(mask_to_rle is None, reason="Module not implemented")
    def test_mask_to_rle_empty(self):
        """Empty mask should encode to valid RLE."""
        mask = np.zeros((8, 8), dtype=np.uint8)
        rle = mask_to_rle(mask)

        assert rle["size"] == [8, 8]
        # Decoded should be all zeros
        decoded = rle_to_mask(rle)
        assert np.all(decoded == 0)


class TestCOCOInstanceExporter:
    """Tests for COCO format exporter."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_tile_data(self, temp_dir):
        """Create sample tile data with image, instance mask, and labels."""
        # Create sample image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image_path = temp_dir / "tile_0000.png"
        cv2.imwrite(str(image_path), image)

        # Create instance mask with 3 instances
        instance_mask = np.zeros((256, 256), dtype=np.uint8)
        # Instance 1: horizontal line at y=50
        instance_mask[48:52, 50:200] = 1
        # Instance 2: horizontal line at y=100
        instance_mask[98:102, 30:220] = 2
        # Instance 3: horizontal line at y=150
        instance_mask[148:152, 60:180] = 3

        instance_mask_path = temp_dir / "tile_0000_instance_mask.png"
        cv2.imwrite(str(instance_mask_path), instance_mask)

        # Create labels JSON
        labels_data = {
            "image": {"path": "tile_0000.png", "size": {"height": 256, "width": 256}},
            "mask": {"path": "tile_0000_mask.png"},
            "instance_mask": {
                "path": "tile_0000_instance_mask.png",
                "instances": [
                    {"instance_id": 1, "elevation": 100},
                    {"instance_id": 2, "elevation": 110},
                    {"instance_id": 3, "elevation": 120},
                ],
            },
            "labels": [],
        }
        labels_path = temp_dir / "tile_0000_labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels_data, f)

        return temp_dir

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_initialization(self, temp_dir):
        """Exporter should initialize with output path."""
        exporter = COCOInstanceExporter(output_path=temp_dir / "coco.json")
        assert exporter.output_path == temp_dir / "coco.json"

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_adds_category(self, temp_dir):
        """Exporter should have contour_line category."""
        exporter = COCOInstanceExporter(output_path=temp_dir / "coco.json")

        coco_data = exporter.to_dict()

        assert len(coco_data["categories"]) == 1
        assert coco_data["categories"][0]["name"] == "contour_line"
        assert coco_data["categories"][0]["id"] == 1

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_add_image(self, temp_dir, sample_tile_data):
        """Exporter should correctly add images."""
        exporter = COCOInstanceExporter(output_path=temp_dir / "coco.json")

        exporter.add_tile(sample_tile_data / "tile_0000_labels.json")
        coco_data = exporter.to_dict()

        assert len(coco_data["images"]) == 1
        assert coco_data["images"][0]["file_name"] == "tile_0000.png"
        assert coco_data["images"][0]["height"] == 256
        assert coco_data["images"][0]["width"] == 256

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_add_annotations(self, temp_dir, sample_tile_data):
        """Exporter should create annotations for each instance."""
        exporter = COCOInstanceExporter(output_path=temp_dir / "coco.json")

        exporter.add_tile(sample_tile_data / "tile_0000_labels.json")
        coco_data = exporter.to_dict()

        # Should have 3 annotations (one per instance)
        assert len(coco_data["annotations"]) == 3

        for ann in coco_data["annotations"]:
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert ann["category_id"] == 1  # contour_line
            assert "segmentation" in ann
            assert "area" in ann
            assert "bbox" in ann
            assert "iscrowd" in ann
            assert ann["iscrowd"] == 0

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_annotation_has_rle_segmentation(self, temp_dir, sample_tile_data):
        """Annotations should have RLE segmentation format."""
        exporter = COCOInstanceExporter(output_path=temp_dir / "coco.json")

        exporter.add_tile(sample_tile_data / "tile_0000_labels.json")
        coco_data = exporter.to_dict()

        for ann in coco_data["annotations"]:
            seg = ann["segmentation"]
            assert "counts" in seg
            assert "size" in seg

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_preserves_elevation(self, temp_dir, sample_tile_data):
        """Annotations should preserve elevation as custom field."""
        exporter = COCOInstanceExporter(output_path=temp_dir / "coco.json")

        exporter.add_tile(sample_tile_data / "tile_0000_labels.json")
        coco_data = exporter.to_dict()

        elevations = sorted([ann.get("elevation") for ann in coco_data["annotations"]])
        assert elevations == [100, 110, 120]

    @pytest.mark.skipif(COCOInstanceExporter is None, reason="Module not implemented")
    def test_exporter_save_json(self, temp_dir, sample_tile_data):
        """Exporter should save valid JSON file."""
        output_path = temp_dir / "coco.json"
        exporter = COCOInstanceExporter(output_path=output_path)

        exporter.add_tile(sample_tile_data / "tile_0000_labels.json")
        exporter.save()

        assert output_path.exists()

        # Verify it's valid JSON
        with open(output_path) as f:
            loaded = json.load(f)

        assert "images" in loaded
        assert "annotations" in loaded
        assert "categories" in loaded


class TestConvertTilesToCoco:
    """Tests for the convenience function to convert a tiles directory."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tiles_directory(self, temp_dir):
        """Create a directory with multiple sample tiles."""
        tiles_dir = temp_dir / "tiles"
        tiles_dir.mkdir()

        for i in range(3):
            # Create instance mask
            instance_mask = np.zeros((128, 128), dtype=np.uint8)
            instance_mask[30:35, 20:100] = 1
            instance_mask[60:65, 20:100] = 2

            cv2.imwrite(
                str(tiles_dir / f"tile_{i:04d}.png"),
                np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                str(tiles_dir / f"tile_{i:04d}_instance_mask.png"), instance_mask
            )

            labels_data = {
                "image": {
                    "path": f"tile_{i:04d}.png",
                    "size": {"height": 128, "width": 128},
                },
                "instance_mask": {
                    "path": f"tile_{i:04d}_instance_mask.png",
                    "instances": [
                        {"instance_id": 1, "elevation": 100 + i * 10},
                        {"instance_id": 2, "elevation": 110 + i * 10},
                    ],
                },
                "labels": [],
            }
            with open(tiles_dir / f"tile_{i:04d}_labels.json", "w") as f:
                json.dump(labels_data, f)

        return tiles_dir

    @pytest.mark.skipif(convert_tiles_to_coco is None, reason="Module not implemented")
    def test_convert_tiles_creates_coco_file(self, tiles_directory, temp_dir):
        """Converting tiles directory should create COCO JSON file."""
        output_path = temp_dir / "coco_annotations.json"

        convert_tiles_to_coco(tiles_directory, output_path)

        assert output_path.exists()

    @pytest.mark.skipif(convert_tiles_to_coco is None, reason="Module not implemented")
    def test_convert_tiles_includes_all_images(self, tiles_directory, temp_dir):
        """All tiles should be included in COCO output."""
        output_path = temp_dir / "coco_annotations.json"

        convert_tiles_to_coco(tiles_directory, output_path)

        with open(output_path) as f:
            coco_data = json.load(f)

        assert len(coco_data["images"]) == 3

    @pytest.mark.skipif(convert_tiles_to_coco is None, reason="Module not implemented")
    def test_convert_tiles_includes_all_annotations(self, tiles_directory, temp_dir):
        """All instances from all tiles should be annotated."""
        output_path = temp_dir / "coco_annotations.json"

        convert_tiles_to_coco(tiles_directory, output_path)

        with open(output_path) as f:
            coco_data = json.load(f)

        # 3 tiles x 2 instances each = 6 annotations
        assert len(coco_data["annotations"]) == 6
