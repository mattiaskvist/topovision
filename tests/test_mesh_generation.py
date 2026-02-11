"""Test script for mesh generation."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from height_extraction.inference import (
    build_adjacency_graph,
    infer_missing_heights,
)
from height_extraction.mesh_generation import export_to_obj, generate_heightmap
from height_extraction.schemas import ContourLine, HeightExtractionOutput


def test_mesh_generation():
    print("Testing mesh generation...")

    # Create synthetic data: a simple pyramid/hill
    # 100x100 image area

    contours = []

    # Base: square at height 0
    contours.append(
        ContourLine(
            id=1,
            points=[(10, 10), (90, 10), (90, 90), (10, 90), (10, 10)],
            height=0.0,
            source="ocr",
        )
    )

    # Middle: square at height 50
    contours.append(
        ContourLine(
            id=2,
            points=[(30, 30), (70, 30), (70, 70), (30, 70), (30, 30)],
            height=50.0,
            source="ocr",
        )
    )

    # Top: point at height 100
    contours.append(ContourLine(id=3, points=[(50, 50)], height=100.0, source="ocr"))

    output = HeightExtractionOutput(image_path="synthetic", contours=contours)

    # Generate mesh
    # Low resolution for speed
    grid_x, grid_y, grid_z = generate_heightmap(output, resolution_scale=0.2)

    # Assertions for grid generation
    assert grid_x.shape == grid_y.shape == grid_z.shape
    assert grid_x.shape[0] > 0 and grid_x.shape[1] > 0

    # Check Z range (should be between 0 and 100, allowing for some interpolation
    # overshoot). Cubic interpolation can overshoot, so we allow a small margin or
    # check approximate range
    assert grid_z.min() >= -10.0  # Allow some overshoot
    assert grid_z.max() <= 110.0

    # Check that the peak is roughly around 100
    assert grid_z.max() > 90.0

    print(f"Generated grid with shape {grid_x.shape}")
    print(f"Z range: {grid_z.min():.2f} to {grid_z.max():.2f}")

    # Export
    output_path = "test_mesh.obj"
    export_to_obj(grid_x, grid_y, grid_z, output_path)

    assert os.path.exists(output_path), "Output file was not created"

    size = os.path.getsize(output_path)
    assert size > 0, "Output file is empty"
    print(f"Successfully created {output_path} ({size} bytes)")

    # Check content
    with open(output_path) as f:
        lines = f.readlines()
        v_count = sum(1 for line in lines if line.startswith("v "))
        f_count = sum(1 for line in lines if line.startswith("f "))

        # Expected vertices: w * h
        w, h = grid_x.shape
        expected_v_count = w * h
        assert v_count == expected_v_count, (
            f"Expected {expected_v_count} vertices, got {v_count}"
        )

        # Expected faces: (w-1) * (h-1) * 2 (since we split quads into 2 triangles)
        expected_f_count = (w - 1) * (h - 1) * 2
        assert f_count == expected_f_count, (
            f"Expected {expected_f_count} faces, got {f_count}"
        )

        print(f"Mesh has {v_count} vertices and {f_count} faces")

    # Cleanup
    os.remove(output_path)


if __name__ == "__main__":
    test_mesh_generation()


def test_mesh_generation_with_inferred_heights(tmp_path):
    print("Testing mesh generation with inferred heights...")

    contours_np = [
        np.array([[[0, 10]], [[40, 10]]], dtype=np.int32),
        np.array([[[0, 20]], [[40, 20]]], dtype=np.int32),
        np.array([[[0, 30]], [[40, 30]]], dtype=np.int32),
    ]
    known_heights = {0: 0.0, 2: 100.0}

    adjacency = build_adjacency_graph(contours_np, (50, 50), max_dist=15.0)
    inferred = infer_missing_heights(contours_np, known_heights, adjacency)

    assert inferred[1] == 50.0

    contours = [
        ContourLine(
            id=1,
            points=[(0, 10), (40, 10)],
            height=inferred.get(0),
            source="ocr",
        ),
        ContourLine(
            id=2,
            points=[(0, 20), (40, 20)],
            height=inferred.get(1),
            source="inference",
        ),
        ContourLine(
            id=3,
            points=[(0, 30), (40, 30)],
            height=inferred.get(2),
            source="ocr",
        ),
    ]

    output = HeightExtractionOutput(image_path="synthetic_missing", contours=contours)
    grid_x, grid_y, grid_z = generate_heightmap(output, resolution_scale=0.5)

    assert grid_x.shape == grid_y.shape == grid_z.shape
    assert grid_z.min() >= -10.0
    assert grid_z.max() <= 110.0

    output_path = tmp_path / "missing_mesh.obj"
    export_to_obj(grid_x, grid_y, grid_z, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_heightmap_clamps_overshoot():
    contours = [
        ContourLine(id=1, points=[(0, 0)], height=0.0, source="ocr"),
        ContourLine(id=2, points=[(0, 40)], height=100.0, source="ocr"),
        ContourLine(id=3, points=[(40, 0)], height=100.0, source="ocr"),
        ContourLine(id=4, points=[(40, 40)], height=0.0, source="ocr"),
    ]
    output = HeightExtractionOutput(image_path="clamp", contours=contours)
    _, _, grid_z = generate_heightmap(
        output, resolution_scale=0.5, interpolation_method="cubic"
    )

    assert not np.isnan(grid_z).any()
    assert grid_z.min() >= 0.0
    assert grid_z.max() <= 100.0


def test_heightmap_collinear_fallback():
    contours = [
        ContourLine(id=1, points=[(0, 0)], height=0.0, source="ocr"),
        ContourLine(id=2, points=[(20, 0)], height=50.0, source="ocr"),
        ContourLine(id=3, points=[(40, 0)], height=100.0, source="ocr"),
    ]
    output = HeightExtractionOutput(image_path="collinear", contours=contours)
    _, _, grid_z = generate_heightmap(
        output, resolution_scale=0.5, interpolation_method="cubic"
    )

    assert not np.isnan(grid_z).any()
    assert grid_z.min() >= 0.0
    assert grid_z.max() <= 100.0


def test_heightmap_rejects_invalid_resolution():
    contours = [
        ContourLine(id=1, points=[(0, 0)], height=0.0, source="ocr"),
        ContourLine(id=2, points=[(40, 40)], height=100.0, source="ocr"),
    ]
    output = HeightExtractionOutput(image_path="invalid", contours=contours)

    with pytest.raises(ValueError, match="resolution_scale"):
        generate_heightmap(output, resolution_scale=0)
