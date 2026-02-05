"""Module for matching OCR results to contour lines."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from OCR.engine.ocr_engine import DetectionResult


def calculate_centroid(polygon_points: list[tuple[int, int]]) -> tuple[float, float]:
    """Calculates the centroid of a polygon.

    Args:
        polygon_points: List of (x, y) coordinates defining the polygon.

    Returns:
        A tuple (x, y) representing the centroid coordinates.

    Raises:
        ValueError: If polygon_points is empty.
    """
    if not polygon_points:
        raise ValueError("polygon_points must not be empty")

    x_coords = [p[0] for p in polygon_points]
    y_coords = [p[1] for p in polygon_points]
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)


def point_line_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculates the minimum distance from a point to a line segment.

    Args:
        px: X-coordinate of the point.
        py: Y-coordinate of the point.
        x1: X-coordinate of the first segment endpoint.
        y1: Y-coordinate of the first segment endpoint.
        x2: X-coordinate of the second segment endpoint.
        y2: Y-coordinate of the second segment endpoint.

    Returns:
        The minimum distance from the point to the line segment.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    # Project point onto line (parameter t)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    t = max(0, min(1, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.hypot(px - closest_x, py - closest_y)


def calculate_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculates the angle of the vector p1->p2 in degrees.

    Args:
        p1: The starting point (x, y).
        p2: The ending point (x, y).

    Returns:
        The angle in degrees.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def calculate_polygon_angle(points: list[tuple[int, int]]) -> float:
    """Calculates a dominant angle for a polygon based on its longest edge.

    Args:
        points: Polygon points in order.

    Returns:
        The angle in degrees of the longest edge, or 0.0 if unavailable.
    """
    if len(points) < 2:
        return 0.0

    best_len = -1.0
    best_angle = 0.0
    n_points = len(points)
    for i in range(n_points):
        p1 = points[i]
        p2 = points[(i + 1) % n_points]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length > best_len:
            best_len = length
            best_angle = calculate_angle(p1, p2)

    return best_angle


def min_distance_to_contour(
    point: tuple[float, float], contour: np.ndarray
) -> tuple[float, float]:
    """Calculates the minimum distance from a point to a contour and the tangent angle.

    Args:
        point: The point (x, y) to measure distance from.
        contour: The contour array of shape (N, 1, 2).

    Returns:
        A tuple containing:
            - The minimum distance from the point to the contour.
            - The angle of the tangent at the closest point on the contour.
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
            best_angle = calculate_angle(tuple(p1), tuple(p2))

    return min_dist, best_angle


def parse_height_text(text: str) -> float | None:
    """Parses height text to float, handling suffixes.

    Args:
        text: The text string containing the height value.

    Returns:
        The parsed height as a float, or None if parsing fails.
    """
    cleaned = "".join(c for c in text if c.isdigit() or c in ".-")
    try:
        return float(cleaned)
    except ValueError:
        return None


def calculate_angle_difference(angle1: float, angle2: float) -> float:
    """Calculates the angular difference accounting for line alignment (modulo 180).

    Since we care about alignment (not direction), angles differing by
    180° are equivalent.

    Args:
        angle1: First angle in degrees.
        angle2: Second angle in degrees.

    Returns:
        The angular difference in degrees, in range [0, 90].
    """
    diff = abs(angle1 - angle2) % 180
    if diff > 90:
        diff = 180 - diff
    return diff


@dataclass
class MatchCandidate:
    """Represents a potential match between a detection and a contour."""

    detection_idx: int
    contour_idx: int
    distance: float
    angle_diff: float
    height: float
    confidence: float

    def cost(self, distance_weight: float = 1.0, angle_weight: float = 1.0) -> float:
        """Calculate weighted cost for this match.

        Lower confidence increases cost (we prefer high-confidence matches).
        """
        base_cost = distance_weight * self.distance + angle_weight * self.angle_diff
        # Scale by inverse confidence: low confidence = higher cost
        confidence_factor = 1.0 / max(self.confidence, 0.01)
        return base_cost * confidence_factor


def compute_match_candidates(
    detections: list[DetectionResult],
    contours: list[np.ndarray],
    max_distance: float = 50.0,
    max_angle_diff: float = 30.0,
    use_angle: bool = False,
) -> list[MatchCandidate]:
    """Compute all valid match candidates between detections and contours.

    Args:
        detections: List of OCR detection results.
        contours: List of contours (numpy arrays of shape (N, 1, 2)).
        max_distance: Maximum distance to consider a match valid.
        max_angle_diff: Maximum angle difference (degrees) to consider valid.
        use_angle: Whether to use angle alignment when filtering candidates.

    Returns:
        List of valid MatchCandidate objects.
    """
    candidates = []

    for det_idx, detection in enumerate(detections):
        height_val = parse_height_text(detection.text)
        if height_val is None:
            continue

        centroid = calculate_centroid(detection.polygon.points)

        text_angle = (
            calculate_polygon_angle(detection.polygon.points) if use_angle else 0.0
        )

        for cont_idx, contour in enumerate(contours):
            dist, contour_angle = min_distance_to_contour(centroid, contour)
            angle_diff = (
                calculate_angle_difference(text_angle, contour_angle)
                if use_angle
                else 0.0
            )

            if dist <= max_distance and (not use_angle or angle_diff <= max_angle_diff):
                candidates.append(
                    MatchCandidate(
                        detection_idx=det_idx,
                        contour_idx=cont_idx,
                        distance=dist,
                        angle_diff=angle_diff,
                        height=height_val,
                        confidence=detection.confidence,
                    )
                )

    return candidates


def build_cost_matrix(
    candidates: list[MatchCandidate],
    n_detections: int,
    n_contours: int,
    distance_weight: float = 1.0,
    angle_weight: float = 1.0,
) -> np.ndarray:
    """Build cost matrix for Hungarian algorithm.

    Args:
        candidates: List of valid match candidates.
        n_detections: Total number of detections.
        n_contours: Total number of contours.
        distance_weight: Weight for distance in cost calculation.
        angle_weight: Weight for angle difference in cost calculation.

    Returns:
        Cost matrix of shape (n_detections, n_contours) with large value
        for invalid pairs.
    """
    # Use a large finite value instead of inf to avoid scipy's "infeasible" error
    # when an entire row/column has no valid candidates.
    invalid_cost = 1e9
    cost_matrix = np.full((n_detections, n_contours), invalid_cost)

    for candidate in candidates:
        cost = candidate.cost(distance_weight, angle_weight)
        # Keep the minimum cost if multiple candidates exist for same pair
        if cost < cost_matrix[candidate.detection_idx, candidate.contour_idx]:
            cost_matrix[candidate.detection_idx, candidate.contour_idx] = cost

    return cost_matrix


def match_text_to_contours(
    detections: list[DetectionResult],
    contours: list[np.ndarray],
    max_distance: float = 50.0,
    max_angle_diff: float = 30.0,
    distance_weight: float = 1.0,
    angle_weight: float = 1.0,
    use_angle: bool = False,
) -> dict[int, float]:
    """Match OCR detections to contours using Hungarian algorithm.

    Uses scipy's linear_sum_assignment to find the globally optimal one-to-one
    matching that minimizes total cost. Cost is based on distance and (optional)
    angle difference, weighted by detection confidence.

    When multiple detections could match the same contour, the one with lowest
    cost (considering confidence) is selected.

    Args:
        detections: List of OCR detection results.
        contours: List of contours (numpy arrays of shape (N, 1, 2)).
        max_distance: Maximum distance to consider a match valid.
        max_angle_diff: Maximum angle difference (degrees) to consider valid.
        distance_weight: Weight for distance in cost calculation.
        angle_weight: Weight for angle difference in cost calculation.
        use_angle: Whether to use angle alignment when filtering candidates.

    Returns:
        Dictionary mapping contour index to matched height value.
    """
    if not detections or not contours:
        return {}

    # Get all valid match candidates
    candidates = compute_match_candidates(
        detections,
        contours,
        max_distance,
        max_angle_diff,
        use_angle,
    )

    if not candidates:
        return {}

    # Build lookup for height values by (detection_idx, contour_idx) pair
    candidate_lookup = {(c.detection_idx, c.contour_idx): c.height for c in candidates}

    # Build cost matrix
    n_detections = len(detections)
    n_contours = len(contours)
    cost_matrix = build_cost_matrix(
        candidates, n_detections, n_contours, distance_weight, angle_weight
    )

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract valid matches (filter out high-cost/invalid assignments)
    # Use the same threshold as build_cost_matrix for invalid pairs
    invalid_cost_threshold = 1e8  # Slightly lower than 1e9
    matches: dict[int, float] = {}
    for det_idx, cont_idx in zip(row_indices, col_indices, strict=False):
        if cost_matrix[det_idx, cont_idx] < invalid_cost_threshold:
            key = (det_idx, cont_idx)
            if key in candidate_lookup:
                matches[cont_idx] = candidate_lookup[key]

    return matches
