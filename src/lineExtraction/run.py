"""Extract lines from binary images as ordered polylines.

Approach:
1. Skeletonize to get 1-pixel wide lines
2. Trace paths from endpoints
3. Merge paths whose endpoints fall within a small radius
"""

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


PATH = Path("./data/umask")


def load_binary_image(image_path: str | Path) -> np.ndarray:
    """Load a binary image as 0s and 1s."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return (img > 127).astype(np.uint8)


def get_neighbors(point: tuple[int, int], skeleton: np.ndarray) -> list[tuple[int, int]]:
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


def trace_path(
    start: tuple[int, int],
    skeleton: np.ndarray,
    visited: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Trace a path from a starting point along the skeleton."""
    path = [start]
    visited.add(start)
    current = start

    while True:
        neighbors = get_neighbors(current, skeleton)
        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            break
        current = unvisited[0]
        path.append(current)
        visited.add(current)

    return path


def extract_paths(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Extract all paths from a skeleton."""
    skeleton_points = set(zip(*np.where(skeleton)))
    if not skeleton_points:
        return []

    # Find endpoints (points with exactly 1 neighbor)
    endpoints = [p for p in skeleton_points if len(get_neighbors(p, skeleton)) == 1]

    paths = []
    visited = set()

    # Start from endpoints
    for ep in endpoints:
        if ep not in visited:
            path = trace_path(ep, skeleton, visited)
            if len(path) >= 2:
                paths.append(path)

    # Handle closed loops (no endpoints)
    remaining = skeleton_points - visited
    while remaining:
        start = remaining.pop()
        path = trace_path(start, skeleton, visited)
        if len(path) >= 2:
            paths.append(path)
        remaining = skeleton_points - visited

    return paths


def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def merge_paths(
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

                # Check all endpoint pairs
                endpoints = [
                    (pi[0], pj[0], True, True),    # start-start
                    (pi[0], pj[-1], True, False),  # start-end
                    (pi[-1], pj[0], False, True),  # end-start
                    (pi[-1], pj[-1], False, False),  # end-end
                ]

                for p1, p2, i_start, j_start in endpoints:
                    if distance(p1, p2) <= radius:
                        # Merge: orient paths so they connect properly
                        if i_start:
                            pi = pi[::-1]
                        if not j_start:
                            pj = pj[::-1]
                        
                        paths[i] = pi + pj
                        paths.pop(j)
                        merged = True
                        break

    return paths


def simplify_path(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    """Simplify path using Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points
    pts = np.array([(p[1], p[0]) for p in points], dtype=np.float32)
    simplified = cv2.approxPolyDP(pts, epsilon, closed=False)
    return [(int(p[0][1]), int(p[0][0])) for p in simplified]


def estimate_line_thickness(binary: np.ndarray) -> float:
    """Estimate average line thickness using distance transform."""
    if binary.sum() == 0:
        return 1.0
    dist = ndimage.distance_transform_edt(binary)
    skel = skeletonize(binary)
    vals = dist[skel > 0]
    if len(vals) == 0:
        return 1.0
    return float(np.median(vals) * 2)


def extract_lines(
    image_path: str | Path,
    simplify: bool = True,
    epsilon: float = 2.0,
) -> list[list[tuple[int, int]]]:
    """Extract lines from a binary image.
    
    Args:
        image_path: Path to binary image (black background, white lines).
        simplify: Apply Douglas-Peucker simplification.
        epsilon: Simplification tolerance.
    
    Returns:
        List of polylines as (row, col) point sequences.
    """
    binary = load_binary_image(image_path)
    
    # Merge radius based on line thickness
    thickness = estimate_line_thickness(binary)
    radius = max(thickness * 2, 10.0)
    
    skeleton = skeletonize(binary).astype(np.uint8)
    paths = extract_paths(skeleton)
    paths = merge_paths(paths, radius)
    
    if simplify:
        paths = [simplify_path(p, epsilon) for p in paths]
    
    return paths


def lines_to_xy(lines: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Convert from (row, col) to (x, y) format."""
    return [[(c, r) for r, c in line] for line in lines]


def visualize(
    original: np.ndarray,
    skeleton: np.ndarray,
    lines: list[list[tuple[int, int]]],
    output_path: str | Path | None = None,
) -> None:
    """Show extraction results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(skeleton, cmap="gray")
    axes[1].set_title("Skeleton")
    axes[1].axis("off")

    axes[2].imshow(np.zeros_like(original), cmap="gray")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(lines), 1)))
    for i, line in enumerate(lines):
        if line:
            rows, cols = zip(*line)
            axes[2].plot(cols, rows, color=colors[i % len(colors)], linewidth=1)
    axes[2].set_title(f"Lines ({len(lines)})")
    axes[2].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    for mask_file in sorted(PATH.glob("*_mask.png")):
        print(f"\n{mask_file.name}")
        
        lines = extract_lines(mask_file)
        print(f"  {len(lines)} lines, {sum(len(l) for l in lines)} points")
        
        binary = load_binary_image(mask_file)
        skeleton = skeletonize(binary).astype(np.uint8)
        visualize(binary, skeleton, lines)


