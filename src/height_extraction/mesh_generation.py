"""Module for generating 3D meshes from height extraction output."""

import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import QhullError

from .schemas import HeightExtractionOutput


def generate_heightmap(
    output: HeightExtractionOutput,
    resolution_scale: float = 1.0,
    interpolation_method: str = "cubic",
    smoothing_sigma: float = 0.0,
    clamp_heights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a dense heightmap from sparse contour lines using interpolation.

    Args:
        output: The HeightExtractionOutput containing contours with heights.
        resolution_scale: Scale factor for the output grid resolution relative to
            the image size. 1.0 means same resolution as image.
        interpolation_method: Method for interpolation ('linear', 'nearest', 'cubic').
        smoothing_sigma: Standard deviation for Gaussian kernel smoothing. 0 to disable.
        clamp_heights: Clamp interpolated heights to the min/max of known values.

    Returns:
        Tuple of (grid_x, grid_y, grid_z) arrays.
    """
    if resolution_scale <= 0:
        raise ValueError("resolution_scale must be > 0")

    valid_methods = {"linear", "nearest", "cubic"}
    if interpolation_method not in valid_methods:
        raise ValueError(f"interpolation_method must be one of {sorted(valid_methods)}")

    # 1. Collect all points with known heights
    points = []
    values = []

    # We need image dimensions. We can infer them from the max coordinates if not
    # available, but ideally we should read the image. For now, let's find the
    # bounds from contours.
    max_x, max_y = 0, 0

    for contour in output.contours:
        if contour.height is not None:
            for x, y in contour.points:
                points.append((x, y))
                values.append(contour.height)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # Update bounds even for contours without height (though we don't use them for
        # interpolation)
        for x, y in contour.points:
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if not points:
        raise ValueError("No contours with assigned heights found in the output.")

    points = np.array(points)
    values = np.array(values)

    method = interpolation_method
    if points.shape[0] < 3:
        method = "nearest"
    else:
        centered = points - points.mean(axis=0)
        rank = np.linalg.matrix_rank(centered)
        if rank < 2:
            method = "nearest"
        elif interpolation_method == "cubic" and points.shape[0] < 4:
            method = "linear"

    if method != interpolation_method:
        print(
            f"Interpolation method downgraded from {interpolation_method} to {method} "
            "due to insufficient point geometry."
        )

    # 2. Create a regular grid
    # Add a small buffer to ensure we cover the edges
    grid_width = int(max_x * resolution_scale) + 1
    grid_height = int(max_y * resolution_scale) + 1

    # Create grid coordinates
    grid_x, grid_y = np.mgrid[0:grid_width, 0:grid_height]

    # Scale grid coordinates back to original image space for interpolation
    # If resolution_scale is 0.5, a grid point at (10, 10) corresponds to (20, 20) in
    # image
    query_points = np.column_stack(
        (grid_x.flatten() / resolution_scale, grid_y.flatten() / resolution_scale)
    )

    # 3. Interpolate
    # griddata expects points as (N, D) and values as (N,)
    # We want to interpolate at query_points
    try:
        grid_z = griddata(points, values, query_points, method=method)
    except QhullError as exc:
        if method == "nearest":
            raise
        print(f"Interpolation failed with {method} ({exc}); falling back to nearest.")
        grid_z = griddata(points, values, query_points, method="nearest")

    # Fill NaNs (outside convex hull of points) with nearest neighbor or min value
    # For a landscape, nearest might be better than 0.
    if np.isnan(grid_z).any():
        grid_z_nearest = griddata(points, values, query_points, method="nearest")
        grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

    grid_z = grid_z.reshape((grid_width, grid_height))

    if smoothing_sigma > 0:
        grid_z = gaussian_filter(grid_z, sigma=smoothing_sigma)

    if clamp_heights:
        min_height = float(values.min())
        max_height = float(values.max())
        grid_z = np.clip(grid_z, min_height, max_height)

    return grid_x, grid_y, grid_z


def export_to_obj(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    output_path: str,
    scale_z: float = 1.0,
):
    """Exports the heightmap grid to a Wavefront .obj file.

    Args:
        grid_x: X coordinates of the grid.
        grid_y: Y coordinates of the grid.
        grid_z: Z coordinates (heights) of the grid.
        output_path: Path to save the .obj file.
        scale_z: Multiplier for Z values to exaggerate relief or correct units.
    """
    grid_width, grid_height = grid_x.shape

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Generated by Topovision\n")
        f.write("o Landscape\n")

        # Write vertices
        # OBJ format: v x y z
        for i in range(grid_width):
            for j in range(grid_height):
                x = grid_x[i, j]
                y = grid_z[i, j] * scale_z
                z = -grid_y[i, j]  # Flip Z to match standard 3D orientation
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")

        # Write faces
        # OBJ indices are 1-based
        # Grid is w x h vertices.
        # Vertex index at (i, j) is i * h + j + 1

        for i in range(grid_width - 1):
            for j in range(grid_height - 1):
                # Define quad: (i, j), (i+1, j), (i+1, j+1), (i, j+1)
                # But we need to map to linear indices

                v1 = i * grid_height + j + 1
                v2 = (i + 1) * grid_height + j + 1
                v3 = (i + 1) * grid_height + (j + 1) + 1
                v4 = i * grid_height + (j + 1) + 1

                # Split quad into two triangles: (v1, v2, v3) and (v1, v3, v4)
                # Reverted order to point normals up
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v1} {v3} {v4}\n")

    print(f"Saved 3D mesh to {output_path}")


def visualize_mesh(mesh_path: str):
    """Visualizes a 3D mesh using Open3D.

    Args:
        mesh_path: Path to the .obj file.
    """
    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertices():
        print("Mesh is empty or could not be loaded.")
        return

    print("Computing normals...")
    mesh.compute_vertex_normals()

    print("Opening visualization window...")
    o3d.visualization.draw_geometries([mesh], window_name="Topovision 3D Viewer")
