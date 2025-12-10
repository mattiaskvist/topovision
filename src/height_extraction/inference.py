"""Module for inferring missing heights using graph adjacency and interpolation."""

import math

import cv2
import numpy as np

from .matcher import min_distance_to_contour


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
    """Infers missing heights based on known heights and adjacency.

    Strategy:
    1. Identify connected components.
    2. For each component, find 'gradients' (pairs of neighbors with known heights).
    3. Determine the contour interval.
    4. Interpolate/Extrapolate.
    """
    inferred = known_heights.copy()

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
    # We use a queue for BFS propagation
    _queue = list(known_heights.keys())
    _visited = set(known_heights.keys())

    # This simple BFS assumes we know the direction (up/down).
    # But we don't. We only know the interval.
    # We need to anchor it.

    # Better approach: Linear Interpolation between knowns.
    # Find paths between known nodes.

    # Let's try a relaxation approach?
    # Or just simple: if a node has TWO known neighbors, check if it fits in between.

    # Let's implement a "Gradient Flow" approach.
    # If we have A(100) -- B(?) -- C(120), then B is likely 110.

    # Iterative pass:
    changed = True
    while changed:
        changed = False
        sorted_nodes = sorted(list(adjacency.keys()))  # Deterministic order

        for i in sorted_nodes:
            if i in inferred:
                continue

            neighbors = list(adjacency[i])
            known_neighbors = [n for n in neighbors if n in inferred]

            if len(known_neighbors) >= 2:
                # Interpolate
                # If we have neighbors with 100 and 120, we are likely 110.
                vals = [inferred[n] for n in known_neighbors]
                avg = sum(vals) / len(vals)

                # Snap to nearest interval multiple?
                # E.g. if avg is 110 and interval is 10, keep 110.
                # If avg is 105, maybe 100 or 110?
                # For now, just take average.
                inferred[i] = avg
                changed = True

            elif len(known_neighbors) == 1:
                # Extrapolate?
                # Only if we have a "direction" from a previous node.
                # This is risky without more info.
                # But if we have A(100) -- B(110) -- C(?), C is likely 120.
                n = known_neighbors[0]
                n_neighbors = adjacency[n]
                n_known_neighbors = [
                    nn for nn in n_neighbors if nn in inferred and nn != i
                ]

                if n_known_neighbors:
                    # We have a chain: nn -> n -> i
                    # Gradient = inferred[n] - inferred[nn]
                    # inferred[i] = inferred[n] + Gradient
                    # But we need to be careful about branching.
                    # Let's just take the first one for now.
                    nn = n_known_neighbors[0]
                    gradient = inferred[n] - inferred[nn]

                    # Limit gradient to reasonable values (e.g. +/- interval)
                    if abs(gradient) > interval * 1.5:
                        # Maybe a jump, or maybe just far away.
                        # Normalize to interval
                        gradient = math.copysign(interval, gradient)

                    inferred[i] = inferred[n] + gradient
                    changed = True

    return inferred
