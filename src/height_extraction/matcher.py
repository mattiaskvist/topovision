"""Module for matching OCR results to contour lines."""

import math

import numpy as np

from OCR.engine.ocr_engine import DetectionResult


def calculate_centroid(polygon_points: list[tuple[int, int]]) -> tuple[float, float]:
    """Calculates the centroid of a polygon."""
    x_coords = [p[0] for p in polygon_points]
    y_coords = [p[1] for p in polygon_points]
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)


def point_line_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculates the minimum distance from a point (px, py) to a line segment."""
    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    # Project point onto line (parameter t)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # Clamp t to segment [0, 1]
    t = max(0, min(1, t))

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.hypot(px - closest_x, py - closest_y)


def calculate_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculates the angle of the vector p1->p2 in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def min_distance_to_contour(
    point: tuple[float, float], contour: np.ndarray
) -> tuple[float, float]:
    """Calculates the minimum distance from a point to a contour and the tangent angle.

    Returns:
        Tuple of (min_distance, tangent_angle_degrees)
    """
    px, py = point
    min_dist = float("inf")
    best_angle = 0.0

    # Contour shape is (N, 1, 2)
    pts = contour[:, 0, :]

    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        dist = point_line_segment_distance(px, py, p1[0], p1[1], p2[0], p2[1])
        if dist < min_dist:
            min_dist = dist
            # Calculate tangent of this segment
            # We normalize angle to [-90, 90] because line direction doesn't matter
            # relative to text (text could be upside down or not)
            # Actually, let's keep full range and handle modulo 180 later
            best_angle = calculate_angle(tuple(p1), tuple(p2))

    return min_dist, best_angle


def parse_height_text(text: str) -> float:
    """Parses height text to float, handling suffixes."""
    # Remove common non-numeric chars except dot and minus
    cleaned = "".join(c for c in text if c.isdigit() or c in ".-")
    try:
        return float(cleaned)
    except ValueError:
        return None


def match_text_to_contours(
    detections: list[DetectionResult],
    contours: list[np.ndarray],
    max_distance: float = 50.0,
    max_angle_diff: float = 30.0,
) -> dict[int, float]:
    """Matches OCR detections to the nearest contour lines with orientation check.

    Args:
        detections: List of OCR detection results.
        contours: List of contours (numpy arrays).
        max_distance: Maximum distance to consider a match valid.
        max_angle_diff: Maximum angle difference (degrees) to consider a match valid.

    Returns:
        Dictionary mapping contour index to matched height value.
    """
    matches = {}  # contour_index -> height

    for detection in detections:
        height_val = parse_height_text(detection.text)
        if height_val is None:
            continue

        centroid = calculate_centroid(detection.polygon.points)

        # Calculate text orientation from top edge (p0 -> p1)
        # Polygon points are usually [top-left, top-right, bottom-right, bottom-left]
        p0 = detection.polygon.points[0]
        p1 = detection.polygon.points[1]
        text_angle = calculate_angle(p0, p1)

        best_contour_idx = -1
        min_dist = float("inf")

        for idx, contour in enumerate(contours):
            dist, contour_angle = min_distance_to_contour(centroid, contour)

            # Check angle difference
            # We care about alignment, so modulo 180
            diff = abs(text_angle - contour_angle) % 180
            if diff > 90:
                diff = 180 - diff

            if dist < min_dist and diff <= max_angle_diff:
                min_dist = dist
                best_contour_idx = idx

        if best_contour_idx != -1 and min_dist <= max_distance:
            # If multiple texts match the same contour, we could average them or
            # take the closest.
            # For now, let's just overwrite (or maybe check consistency later).
            matches[best_contour_idx] = height_val

    return matches
