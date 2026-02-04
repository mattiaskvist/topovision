"""Extract contour lines from binary images as ordered polylines.

Approach:
1. Skeletonize to get 1-pixel wide lines
2. Trace paths from endpoints
3. Merge paths whose endpoints fall within a small radius
4. Simplify using Douglas-Peucker algorithm
"""

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from src.height_extraction.schemas import ContourLine


def _get_neighbors(point: tuple[int, int], skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Get 8-connected skeleton neighbors of a point."""
    row, col = point
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]:
                if skeleton[nr, nc]:
                    neighbors.append((nr, nc))
    return neighbors


def _trace_path(
    start: tuple[int, int],
    skeleton: np.ndarray,
    visited: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Trace a path from a starting point along the skeleton."""
    path = [start]
    visited.add(start)
    current = start

    while True:
        neighbors = _get_neighbors(current, skeleton)
        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            break
        current = unvisited[0]
        path.append(current)
        visited.add(current)

    return path


def _extract_paths(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Extract all paths from a skeleton image."""
    skeleton_points = set(zip(*np.where(skeleton)))
    if not skeleton_points:
        return []

    endpoints = [p for p in skeleton_points if len(_get_neighbors(p, skeleton)) == 1]
    paths = []
    visited: set[tuple[int, int]] = set()

    for ep in endpoints:
        if ep not in visited:
            path = _trace_path(ep, skeleton, visited)
            if len(path) >= 2:
                paths.append(path)

    # Handle closed loops (no endpoints)
    remaining = skeleton_points - visited
    while remaining:
        start = remaining.pop()
        path = _trace_path(start, skeleton, visited)
        if len(path) >= 2:
            paths.append(path)
        remaining = skeleton_points - visited

    return paths


def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _merge_paths(
    paths: list[list[tuple[int, int]]],
    radius: float,
) -> list[list[tuple[int, int]]]:
    """Merge paths whose endpoints are within a given radius."""
    if len(paths) <= 1:
        return paths

    paths = [list(p) for p in paths]
    merged = True

    while merged:
        merged = False
        for i in range(len(paths)):
            if merged:
                break
            for j in range(i + 1, len(paths)):
                if merged:
                    break

                pi, pj = paths[i], paths[j]
                if len(pi) < 2 or len(pj) < 2:
                    continue

                endpoint_pairs = [
                    (pi[0], pj[0], True, True),
                    (pi[0], pj[-1], True, False),
                    (pi[-1], pj[0], False, True),
                    (pi[-1], pj[-1], False, False),
                ]

                for p1, p2, i_start, j_start in endpoint_pairs:
                    if _distance(p1, p2) <= radius:
                        if i_start:
                            pi = pi[::-1]
                        if not j_start:
                            pj = pj[::-1]
                        paths[i] = pi + pj
                        paths.pop(j)
                        merged = True
                        break

    return paths


def _simplify_path(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    """Simplify path using Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points
    pts = np.array([(p[1], p[0]) for p in points], dtype=np.float32)
    simplified = cv2.approxPolyDP(pts, epsilon, closed=False)
    return [(int(p[0][1]), int(p[0][0])) for p in simplified]


def _estimate_line_thickness(binary: np.ndarray) -> float:
    """Estimate average line thickness using distance transform."""
    if binary.sum() == 0:
        return 1.0
    dist = ndimage.distance_transform_edt(binary)
    skel = skeletonize(binary)
    vals = dist[skel > 0]
    if len(vals) == 0:
        return 1.0
    return float(np.median(vals) * 2)


def extract_contours(
    binary_image: np.ndarray,
    simplify: bool = True,
    epsilon: float = 2.0,
) -> list[ContourLine]:
    """Extract contour lines from a binary image.

    Args:
        binary_image: Binary image array (0s and 1s, or 0s and 255s).
        simplify: Apply Douglas-Peucker simplification.
        epsilon: Simplification tolerance.

    Returns:
        List of ContourLine objects with (x, y) point sequences.
    """
    binary = (binary_image > 0).astype(np.uint8)

    thickness = _estimate_line_thickness(binary)
    radius = max(thickness * 2, 10.0)

    skeleton = skeletonize(binary).astype(np.uint8)
    paths = _extract_paths(skeleton)
    paths = _merge_paths(paths, radius)

    if simplify:
        paths = [_simplify_path(p, epsilon) for p in paths]

    return [
        ContourLine(
            id=i,
            points=[(col, row) for row, col in path],
            height=None,
            source="unknown",
        )
        for i, path in enumerate(paths)
    ]


def extract_contours_from_file(
    image_path: str | Path,
    simplify: bool = True,
    epsilon: float = 2.0,
) -> list[ContourLine]:
    """Extract contour lines from a binary image file.

    Args:
        image_path: Path to binary image (black background, white lines).
        simplify: Apply Douglas-Peucker simplification.
        epsilon: Simplification tolerance.

    Returns:
        List of ContourLine objects with (x, y) point sequences.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return extract_contours((img > 127).astype(np.uint8), simplify, epsilon)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PATH = Path("./data/umask")

    for mask_file in sorted(PATH.glob("*_mask.png")):
        print(f"\n{mask_file.name}")

        # Load image and extract contours
        img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        binary = (img > 127).astype(np.uint8)
        contours = extract_contours(binary)
        total_points = sum(len(c.points) for c in contours)
        print(f"  {len(contours)} lines, {total_points} points")

        # Visualize
        skeleton = skeletonize(binary).astype(np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(binary, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(skeleton, cmap="gray")
        axes[1].set_title("Skeleton")
        axes[1].axis("off")

        axes[2].imshow(np.zeros_like(binary), cmap="gray")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(contours), 1)))
        for i, contour in enumerate(contours):
            if contour.points:
                xs, ys = zip(*contour.points)
                axes[2].plot(xs, ys, color=colors[i % len(colors)], linewidth=1)
        axes[2].set_title(f"Lines ({len(contours)})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()
