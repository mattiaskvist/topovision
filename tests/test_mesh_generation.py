"""Test script for mesh generation."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

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

    print(f"Generated grid with shape {grid_x.shape}")
    print(f"Z range: {grid_z.min():.2f} to {grid_z.max():.2f}")

    # Export
    output_path = "test_mesh.obj"
    export_to_obj(grid_x, grid_y, grid_z, output_path)

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"Successfully created {output_path} ({size} bytes)")

        # Check content
        with open(output_path) as f:
            lines = f.readlines()
            v_count = sum(1 for line in lines if line.startswith("v "))
            f_count = sum(1 for line in lines if line.startswith("f "))
            print(f"Mesh has {v_count} vertices and {f_count} faces")

        # Cleanup
        os.remove(output_path)
    else:
        print("Failed to create output file")


if __name__ == "__main__":
    test_mesh_generation()
