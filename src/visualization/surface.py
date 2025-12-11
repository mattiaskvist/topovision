"""Module for interpolating a surface from sparse points."""

import numpy as np
from scipy.interpolate import griddata


def interpolate_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_shape: tuple[int, int],
    method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolates a regular grid surface from sparse points.

    Args:
        x: Array of x coordinates.
        y: Array of y coordinates.
        z: Array of z coordinates.
        grid_shape: Shape of the target grid (height, width).
        method: Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        Tuple of (grid_x, grid_y, grid_z).
    """
    h, w = grid_shape
    grid_x, grid_y = np.mgrid[0:w, 0:h]  # Note: mgrid is [x_start:x_end, y_start:y_end]
    # Actually mgrid[0:h, 0:w] produces indices.
    # Let's use meshgrid for clarity matching image coordinates.

    # Image coordinates: x is column (0..w), y is row (0..h)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    points = np.column_stack((x, y))

    # griddata expects (n, D) points and (n,) values
    # We want to interpolate at (grid_x, grid_y)

    grid_z = griddata(points, z, (grid_x, grid_y), method=method)

    return grid_x, grid_y, grid_z
