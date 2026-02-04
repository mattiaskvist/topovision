"""Tests for tile_generator module with instance segmentation support."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import will work after we implement the module
try:
    from data_generation.tile_generator import (
        InstanceMaskResult,
        render_instance_mask,
        render_mask,
    )
except ImportError:
    # Module not yet implemented, tests will be skipped
    render_mask = None
    render_instance_mask = None
    InstanceMaskResult = None


def create_mock_gdf_with_lines(num_lines: int = 3):
    """Create a mock GeoDataFrame with simple line geometries."""
    import geopandas as gpd
    from shapely.geometry import LineString

    lines = []
    elevations = []
    for i in range(num_lines):
        # Create horizontal lines at different y positions
        y = 0.2 + i * 0.2
        line = LineString([(0.1, y), (0.9, y)])
        lines.append(line)
        elevations.append(100 + i * 10)

    gdf = gpd.GeoDataFrame({"geometry": lines, "elevation": elevations})
    return gdf


class TestRenderMask:
    """Tests for the existing render_mask function (binary mask)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_gdf(self):
        """Create a mock GeoDataFrame with contour lines."""
        return create_mock_gdf_with_lines(3)

    @pytest.mark.skipif(render_mask is None, reason="Module not implemented")
    def test_render_mask_creates_binary_mask(self, temp_dir, mock_gdf):
        """Binary mask should have values 0 and 255 only."""
        import cv2

        mask_path = temp_dir / "test_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        render_mask(mock_gdf, bounds, mask_path, size=256)

        assert mask_path.exists()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique_values = np.unique(mask)
        # Binary mask should only have 0 (background) and 255 (contour)
        assert set(unique_values).issubset({0, 255})

    @pytest.mark.skipif(render_mask is None, reason="Module not implemented")
    def test_render_mask_has_contour_pixels(self, temp_dir, mock_gdf):
        """Binary mask should contain some white pixels for contours."""
        import cv2

        mask_path = temp_dir / "test_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        render_mask(mock_gdf, bounds, mask_path, size=256)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Should have some contour pixels (value 255)
        assert np.sum(mask == 255) > 0


class TestRenderInstanceMask:
    """Tests for the new instance mask generation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_gdf(self):
        """Create a mock GeoDataFrame with contour lines."""
        return create_mock_gdf_with_lines(3)

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_creates_file(self, temp_dir, mock_gdf):
        """Instance mask file should be created."""
        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        result = render_instance_mask(mock_gdf, bounds, mask_path, size=256)

        assert mask_path.exists()
        assert isinstance(result, InstanceMaskResult)

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_unique_ids(self, temp_dir, mock_gdf):
        """Each contour line should have a unique instance ID."""
        import cv2

        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        _result = render_instance_mask(mock_gdf, bounds, mask_path, size=256)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Unique values should be 0 (background) + one per contour line
        unique_values = set(np.unique(mask))
        unique_values.discard(0)  # Remove background
        # Should have one unique ID per contour line (3 lines)
        assert len(unique_values) == 3

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_returns_instance_info(self, temp_dir, mock_gdf):
        """Result should contain mapping of instance IDs to elevations."""
        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        result = render_instance_mask(mock_gdf, bounds, mask_path, size=256)

        # Should have instance info for each contour
        assert len(result.instances) == 3
        for instance in result.instances:
            assert "instance_id" in instance
            assert "elevation" in instance
            assert instance["instance_id"] > 0  # IDs start from 1

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_consecutive_ids(self, temp_dir, mock_gdf):
        """Instance IDs should be consecutive starting from 1."""
        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        result = render_instance_mask(mock_gdf, bounds, mask_path, size=256)

        instance_ids = [inst["instance_id"] for inst in result.instances]
        assert sorted(instance_ids) == [1, 2, 3]

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_elevation_preserved(self, temp_dir, mock_gdf):
        """Elevation values should be preserved in instance info."""
        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        result = render_instance_mask(mock_gdf, bounds, mask_path, size=256)

        elevations = sorted([inst["elevation"] for inst in result.instances])
        assert elevations == [100, 110, 120]

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_empty_gdf(self, temp_dir):
        """Empty GeoDataFrame should return empty result."""
        import geopandas as gpd

        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)
        empty_gdf = gpd.GeoDataFrame({"geometry": [], "ELEV": []})

        result = render_instance_mask(empty_gdf, bounds, mask_path, size=256)

        assert result is None or len(result.instances) == 0

    @pytest.mark.skipif(render_instance_mask is None, reason="Module not implemented")
    def test_render_instance_mask_max_255_instances(self, temp_dir):
        """Should handle up to 255 instances (uint8 limit)."""
        import geopandas as gpd
        from shapely.geometry import LineString

        # Create 255 lines
        lines = []
        elevations = []
        for i in range(255):
            y = 0.001 + i * 0.003
            line = LineString([(0.1, y), (0.9, y)])
            lines.append(line)
            elevations.append(100 + i)

        gdf = gpd.GeoDataFrame({"geometry": lines, "ELEV": elevations})
        mask_path = temp_dir / "test_instance_mask.png"
        bounds = (0.0, 0.0, 1.0, 1.0)

        result = render_instance_mask(gdf, bounds, mask_path, size=512)

        assert len(result.instances) == 255


class TestInstanceMaskResult:
    """Tests for the InstanceMaskResult data class."""

    @pytest.mark.skipif(InstanceMaskResult is None, reason="Module not implemented")
    def test_instance_mask_result_structure(self):
        """InstanceMaskResult should have expected attributes."""
        result = InstanceMaskResult(
            mask_path=Path("/tmp/mask.png"),
            instances=[
                {"instance_id": 1, "elevation": 100},
                {"instance_id": 2, "elevation": 110},
            ],
        )

        assert result.mask_path == Path("/tmp/mask.png")
        assert len(result.instances) == 2

    @pytest.mark.skipif(InstanceMaskResult is None, reason="Module not implemented")
    def test_instance_mask_result_to_dict(self):
        """InstanceMaskResult should be convertible to dict for JSON."""
        result = InstanceMaskResult(
            mask_path=Path("/tmp/mask.png"),
            instances=[{"instance_id": 1, "elevation": 100}],
        )

        result_dict = result.to_dict()
        assert "mask_path" in result_dict
        assert "instances" in result_dict
        # Should be JSON serializable
        json.dumps(result_dict)
