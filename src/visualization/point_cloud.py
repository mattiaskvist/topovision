"""Module for converting contours to point cloud data."""

import numpy as np


def contours_to_points(
    contours: list[np.ndarray], height_map: dict[int, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts contours and height map to x, y, z arrays.

    Args:
        contours: List of contours.
        height_map: Dictionary mapping contour index to height.

    Returns:
        Tuple of (x, y, z) numpy arrays.
    """
    xs = []
    ys = []
    zs = []

    for idx, contour in enumerate(contours):
        height = height_map.get(idx)
        if height is not None:
            # contour is (N, 1, 2)
            pts = contour[:, 0, :]
            xs.extend(pts[:, 0])
            ys.extend(pts[:, 1])
            zs.extend([height] * len(pts))

    return np.array(xs), np.array(ys), np.array(zs)
