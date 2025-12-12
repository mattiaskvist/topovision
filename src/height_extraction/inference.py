"""Module for inferring missing heights using graph adjacency and interpolation."""

import math

import cv2
import numpy as np

from .matcher import calculate_centroid, min_distance_to_contour


def build_adjacency_graph(
    contours: list[np.ndarray], image_shape: tuple[int, int], max_dist: float = 20.0
) -> dict[int, set[int]]:
    """Builds adjacency graph with nodes as contour idx and edges for spatial proximity.

    Args:
        contours: List of contours.
        image_shape: Shape of the image (height, width).
        max_dist: Maximum distance between contours to consider them adjacent.

    Returns:
        Adjacency list mapping contour index to a set of adjacent contour indices.
    """
    adj = {i: set() for i in range(len(contours))}

    # This is O(N^2) which is fine for N < ~500.
    # For larger N, we need a spatial index or the dilation method.
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            cnt_i = contours[i]
            cnt_j = contours[j]

            x, y, w, h = cv2.boundingRect(cnt_i)
            x2, y2, w2, h2 = cv2.boundingRect(cnt_j)

            if (
                x > x2 + w2 + max_dist
                or x2 > x + w + max_dist
                or y > y2 + h2 + max_dist
                or y2 > y + h + max_dist
            ):
                continue

            # Detailed check
            step = max(1, len(cnt_i) // 20)
            sample_points = cnt_i[::step]

            dist = float("inf")
            for pt in sample_points:
                d, _ = min_distance_to_contour(tuple(pt[0]), cnt_j)
                if d < dist:
                    dist = d
                if dist < max_dist:
                    break

            if dist < max_dist:
                adj[i].add(j)
                adj[j].add(i)

    return adj


def build_hierarchy(contours: list[np.ndarray]) -> dict[int, int]:
    """Builds a hierarchy of contours to determine nesting.

    Args:
        contours: List of contours.

    Returns:
        Dictionary mapping contour index to its parent contour index (or -1 if none).
    """
    hierarchy = {}
    for i, cnt_i in enumerate(contours):
        parent = -1
        # Find the smallest contour that contains cnt_i
        min_area = float("inf")

        moments = cv2.moments(cnt_i)
        if moments["m00"] == 0:
            cx, cy = 0, 0
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        for j, cnt_j in enumerate(contours):
            if i == j:
                continue

            # pointPolygonTest returns > 0 if inside
            if cv2.pointPolygonTest(cnt_j, (cx, cy), False) > 0:
                area = cv2.contourArea(cnt_j)
                if area < min_area:
                    min_area = area
                    parent = j

        hierarchy[i] = parent
    return hierarchy


def infer_missing_heights(
    contours: list[np.ndarray],
    known_heights: dict[int, float],
    adjacency: dict[int, set[int]],
) -> dict[int, float]:
    """Infers missing heights based on known heights, spatial gradients, and nesting.

    Args:
        contours: List of contours.
        known_heights: Dictionary of known heights.
        adjacency: Adjacency graph of contours.

    Returns:
        Dictionary mapping contour index to inferred height (including known heights).
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

    # 2. Build Hierarchy
    hierarchy = build_hierarchy(contours)

    # 3. Propagation
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
            n = known_neighbors[0]

            # Try Spatial Gradient first
            n_neighbors = adjacency[n]
            n_known_neighbors = [nn for nn in n_neighbors if nn in inferred and nn != i]

            gradient_found = False

            if n_known_neighbors:
                # Use the best aligned neighbor
                best_nn = None
                max_alignment = -1.0

                # Vector n -> i
                vec_ni = np.array(centroids[i]) - np.array(centroids[n])
                norm_ni = np.linalg.norm(vec_ni)
                if norm_ni > 0:
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
                        if gradient != 0:
                            step = math.copysign(interval, gradient)
                            inferred[i] = inferred[n] + step
                            changed = True
                            gradient_found = True

            if gradient_found:
                continue

            # Case 3: Nesting Logic (Hill/Depression)
            # If i is inside n, or n is inside i
            parent_i = hierarchy.get(i, -1)
            parent_n = hierarchy.get(n, -1)

            is_nested = (parent_i == n) or (parent_n == i)

            if is_nested:
                # Check trend of n relative to its neighbors (or parent)
                # If n is higher than its parent/neighbors,
                # and i is inside n -> Hill -> i > n
                # If n is lower -> Depression -> i < n

                # Find a reference for n (parent or another neighbor)
                ref_n = -1
                if parent_n != -1 and parent_n in inferred:
                    ref_n = parent_n
                elif n_known_neighbors:
                    ref_n = n_known_neighbors[0]  # Just pick one

                if ref_n != -1:
                    trend = inferred[n] - inferred[ref_n]
                    if trend > 0:
                        # n is higher than ref -> Rising -> i should be higher
                        inferred[i] = inferred[n] + interval
                    else:
                        # n is lower than ref -> Falling -> i should be lower
                        inferred[i] = inferred[n] - interval
                else:
                    # No reference, assume Hill (rising)
                    # If i is inside n, i > n
                    if parent_i == n:
                        inferred[i] = inferred[n] + interval
                    else:
                        # n is inside i, so i is outer -> i < n
                        inferred[i] = inferred[n] - interval

                changed = True

    return inferred
