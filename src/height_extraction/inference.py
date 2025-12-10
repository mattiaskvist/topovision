"""Module for inferring missing heights using graph adjacency and interpolation."""

import math

import cv2
import numpy as np

from .matcher import calculate_centroid, min_distance_to_contour


def build_adjacency_graph(
    contours: list[np.ndarray], image_shape: tuple[int, int], max_dist: float = 20.0
) -> dict[int, set[int]]:
    """Builds adjacency graph with nodes as contour idx and edges for spatial proximity.

    This uses a simple distance-based approach. A more robust approach might
    use dilation or Voronoi diagrams.
    """
    adj = {i: set() for i in range(len(contours))}

    # This is O(N^2) which is fine for N < ~500.
    # For larger N, we need a spatial index or the dilation method.
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            # Check distance between contour i and j
            # We can check min distance between their points
            # To speed up, check bounding box overlap/distance first

            # Simple check: min distance between any points
            # We can reuse min_distance_to_contour but that takes a point.
            # Let's just take a subsample of points from i and check dist to j.

            cnt_i = contours[i]
            cnt_j = contours[j]

            # Quick bbox check
            x, y, w, h = cv2.boundingRect(cnt_i)
            x2, y2, w2, h2 = cv2.boundingRect(cnt_j)

            # Expand bbox by max_dist
            if (
                x > x2 + w2 + max_dist
                or x2 > x + w + max_dist
                or y > y2 + h2 + max_dist
                or y2 > y + h + max_dist
            ):
                continue

            # Detailed check
            # Sample points from i
            step = max(1, len(cnt_i) // 20)
            sample_points = cnt_i[::step]

            dist = float("inf")
            for pt in sample_points:
                d = min_distance_to_contour(tuple(pt[0]), cnt_j)
                if d < dist:
                    dist = d
                if dist < max_dist:
                    break

            if dist < max_dist:
                adj[i].add(j)
                adj[j].add(i)

    return adj


def infer_missing_heights(
    contours: list[np.ndarray],
    known_heights: dict[int, float],
    adjacency: dict[int, set[int]],
) -> dict[int, float]:
    """Infers missing heights based on known heights and spatial gradients.

    Strategy:
    1. Calculate centroids for all contours.
    2. Determine contour interval.
    3. Propagate heights using spatial gradient:
       - If we know H(A) and H(B), and C is adjacent to B and "in the same direction"
         as A->B, then H(C) should follow the gradient.
    """
    inferred = known_heights.copy()
    centroids = {}
    for idx, cnt in enumerate(contours):
        # cnt is (N, 1, 2)
        points = [(p[0][0], p[0][1]) for p in cnt]
        centroids[idx] = calculate_centroid(points)

    # 1. Find Contour Interval
    intervals = []
    for i, h_i in known_heights.items():
        for j in adjacency[i]:
            if j in known_heights:
                h_j = known_heights[j]
                diff = abs(h_i - h_j)
                if diff > 0:
                    intervals.append(diff)

    interval = 10.0 if not intervals else min(intervals)
    print(f"Estimated Contour Interval: {interval}")

    # 2. Propagation
    changed = True
    while changed:
        changed = False
        sorted_nodes = sorted(list(adjacency.keys()))  # Deterministic order

        for i in sorted_nodes:
            if i in inferred:
                continue

            neighbors = list(adjacency[i])
            known_neighbors = [n for n in neighbors if n in inferred]

            if not known_neighbors:
                continue

            # Case 1: Interpolation (Between two knowns)
            if len(known_neighbors) >= 2:
                vals = [inferred[n] for n in known_neighbors]
                avg = sum(vals) / len(vals)
                # Snap to nearest interval? For now, just avg.
                inferred[i] = avg
                changed = True
                continue

            # Case 2: Extrapolation (Spatial Gradient)
            # We have one known neighbor 'n'.
            # We need a 'prev' neighbor of 'n' to establish a gradient.
            n = known_neighbors[0]
            n_neighbors = adjacency[n]
            n_known_neighbors = [nn for nn in n_neighbors if nn in inferred and nn != i]

            if n_known_neighbors:
                # Use the best aligned neighbor
                best_nn = None
                max_alignment = -1.0

                # Vector n -> i
                vec_ni = np.array(centroids[i]) - np.array(centroids[n])
                norm_ni = np.linalg.norm(vec_ni)
                if norm_ni == 0:
                    continue
                vec_ni /= norm_ni

                for nn in n_known_neighbors:
                    # Vector nn -> n
                    vec_nn_n = np.array(centroids[n]) - np.array(centroids[nn])
                    norm_nn_n = np.linalg.norm(vec_nn_n)
                    if norm_nn_n == 0:
                        continue
                    vec_nn_n /= norm_nn_n

                    # Dot product to check alignment
                    alignment = np.dot(vec_nn_n, vec_ni)

                    # We want alignment close to 1 (straight line)
                    if alignment > max_alignment:
                        max_alignment = alignment
                        best_nn = nn

                # Threshold for alignment (e.g., > 0 means generally same direction)
                if best_nn is not None and max_alignment > 0.5:
                    gradient = inferred[n] - inferred[best_nn]

                    # Normalize gradient magnitude to interval
                    # (Assuming step is always 1 interval)
                    if gradient != 0:
                        step = math.copysign(interval, gradient)
                        inferred[i] = inferred[n] + step
                        changed = True

    return inferred
