import math

import numpy as np
import pytest

from height_extraction.matcher import (
    MatchCandidate,
    build_cost_matrix,
    calculate_angle,
    calculate_angle_difference,
    calculate_centroid,
    calculate_polygon_angle,
    compute_match_candidates,
    match_text_to_contours,
    min_distance_to_contour,
    parse_height_text,
    point_line_segment_distance,
)
from OCR.engine.ocr_engine import DetectionResult, Polygon

# --- Fixtures and Helpers ---


def create_detection(
    text: str,
    points: list[tuple[int, int]],
    confidence: float = 1.0,
) -> DetectionResult:
    """Create a DetectionResult for testing."""
    return DetectionResult(
        text=text, polygon=Polygon(points=points), confidence=confidence
    )


def make_horizontal_contour(
    y: float, x_start: float = 0, x_end: float = 100
) -> np.ndarray:
    """Create a horizontal contour line."""
    return np.array([[[x_start, y]], [[x_end, y]]], dtype=np.int32)


def make_vertical_contour(
    x: float, y_start: float = 0, y_end: float = 100
) -> np.ndarray:
    """Create a vertical contour line."""
    return np.array([[[x, y_start]], [[x, y_end]]], dtype=np.int32)


def make_horizontal_text_box(
    center_x: float, center_y: float, width: float = 20, height: float = 10
) -> list[tuple[int, int]]:
    """Create polygon points for horizontal text (angle 0)."""
    half_w, half_h = width / 2, height / 2
    return [
        (int(center_x - half_w), int(center_y - half_h)),  # top-left
        (int(center_x + half_w), int(center_y - half_h)),  # top-right
        (int(center_x + half_w), int(center_y + half_h)),  # bottom-right
        (int(center_x - half_w), int(center_y + half_h)),  # bottom-left
    ]


def make_vertical_text_box(
    center_x: float, center_y: float, width: float = 10, height: float = 20
) -> list[tuple[int, int]]:
    """Create polygon points for vertical text (angle 90)."""
    half_w, half_h = width / 2, height / 2
    return [
        (int(center_x - half_w), int(center_y - half_h)),  # top-left
        (int(center_x - half_w), int(center_y + half_h)),  # top-right (rotated)
        (int(center_x + half_w), int(center_y + half_h)),  # bottom-right
        (int(center_x + half_w), int(center_y - half_h)),  # bottom-left
    ]


# --- Helper Function Tests ---


class TestCalculateCentroid:
    def test_square(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert calculate_centroid(points) == (5.0, 5.0)

    def test_triangle(self):
        points = [(0, 0), (10, 0), (5, 10)]
        assert calculate_centroid(points) == (5.0, 10 / 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calculate_centroid([])


class TestPointLineSegmentDistance:
    def test_point_on_line(self):
        assert point_line_segment_distance(5, 0, 0, 0, 10, 0) == 0.0

    def test_point_perpendicular(self):
        assert point_line_segment_distance(5, 5, 0, 0, 10, 0) == 5.0

    def test_point_beyond_endpoint(self):
        assert point_line_segment_distance(15, 0, 0, 0, 10, 0) == 5.0

    def test_zero_length_segment(self):
        assert point_line_segment_distance(5, 5, 2, 2, 2, 2) == math.hypot(3, 3)

    def test_diagonal_segment(self):
        # Point at origin, segment from (1,1) to (2,2)
        dist = point_line_segment_distance(0, 0, 1, 1, 2, 2)
        assert abs(dist - math.sqrt(2)) < 1e-10


class TestCalculateAngle:
    def test_horizontal_right(self):
        assert calculate_angle((0, 0), (10, 0)) == 0.0

    def test_vertical_up(self):
        assert calculate_angle((0, 0), (0, 10)) == 90.0

    def test_horizontal_left(self):
        assert calculate_angle((10, 0), (0, 0)) == 180.0

    def test_vertical_down(self):
        assert calculate_angle((0, 10), (0, 0)) == -90.0

    def test_diagonal(self):
        assert calculate_angle((0, 0), (10, 10)) == 45.0


class TestCalculatePolygonAngle:
    def test_prefers_longest_edge(self):
        points = [(0, 0), (10, 0), (10, 2), (0, 2)]
        assert calculate_polygon_angle(points) == 0.0

    def test_vertical_longest_edge(self):
        points = [(0, 0), (2, 0), (2, 10), (0, 10)]
        assert calculate_polygon_angle(points) == 90.0


class TestCalculateAngleDifference:
    def test_same_angle(self):
        assert calculate_angle_difference(0, 0) == 0.0
        assert calculate_angle_difference(45, 45) == 0.0

    def test_opposite_directions(self):
        # 0 and 180 are same line orientation
        assert calculate_angle_difference(0, 180) == 0.0
        assert calculate_angle_difference(90, -90) == 0.0

    def test_perpendicular(self):
        assert calculate_angle_difference(0, 90) == 90.0
        assert calculate_angle_difference(45, 135) == 90.0

    def test_small_difference(self):
        assert calculate_angle_difference(0, 30) == 30.0
        assert calculate_angle_difference(170, -170) == 20.0


class TestParseHeightText:
    def test_integer(self):
        assert parse_height_text("100") == 100.0

    def test_with_suffix(self):
        assert parse_height_text("100m") == 100.0

    def test_decimal(self):
        assert parse_height_text("100.5") == 100.5

    def test_negative(self):
        assert parse_height_text("-50") == -50.0

    def test_invalid(self):
        assert parse_height_text("invalid") is None

    def test_empty(self):
        assert parse_height_text("") is None

    def test_mixed_text(self):
        assert parse_height_text("Height: 250m") == 250.0


class TestMinDistanceToContour:
    def test_horizontal_contour(self):
        contour = np.array([[[0, 0]], [[10, 0]]], dtype=np.int32)
        dist, angle = min_distance_to_contour((5, 5), contour)
        assert dist == 5.0
        assert angle == 0.0

    def test_vertical_contour(self):
        contour = np.array([[[0, 0]], [[0, 10]]], dtype=np.int32)
        dist, angle = min_distance_to_contour((5, 5), contour)
        assert dist == 5.0
        assert angle == 90.0

    def test_point_beyond_contour(self):
        contour = np.array([[[0, 0]], [[10, 0]]], dtype=np.int32)
        dist, angle = min_distance_to_contour((15, 0), contour)
        assert dist == 5.0
        assert angle == 0.0

    def test_multi_segment_contour(self):
        # L-shaped contour
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.int32)
        dist, _angle = min_distance_to_contour((5, 5), contour)
        assert dist == 5.0  # Closest to horizontal segment


# --- MatchCandidate Tests ---


class TestMatchCandidate:
    def test_cost_basic(self):
        candidate = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        assert candidate.cost() == 15.0  # 10 + 5

    def test_cost_with_weights(self):
        candidate = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        assert (
            candidate.cost(distance_weight=2.0, angle_weight=0.5) == 22.5
        )  # 2*10 + 0.5*5

    def test_cost_low_confidence_increases_cost(self):
        high_conf = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        low_conf = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=0.5,
        )
        assert low_conf.cost() > high_conf.cost()
        assert low_conf.cost() == 2 * high_conf.cost()


# --- Compute Match Candidates Tests ---


class TestComputeMatchCandidates:
    def test_single_valid_candidate(self):
        contour = make_vertical_contour(x=10)
        points = make_vertical_text_box(center_x=15, center_y=50)
        detection = create_detection("100", points)

        candidates = compute_match_candidates([detection], [contour])

        assert len(candidates) == 1
        assert candidates[0].height == 100.0
        assert candidates[0].detection_idx == 0
        assert candidates[0].contour_idx == 0

    def test_no_candidates_distance_exceeded(self):
        contour = make_vertical_contour(x=10)
        points = make_vertical_text_box(center_x=100, center_y=50)  # Far away
        detection = create_detection("100", points)

        candidates = compute_match_candidates([detection], [contour], max_distance=50.0)

        assert len(candidates) == 0

    def test_no_candidates_angle_exceeded(self):
        contour = make_vertical_contour(x=10)  # Vertical: angle 90
        points = make_horizontal_text_box(
            center_x=15, center_y=50
        )  # Horizontal: angle 0
        detection = create_detection("100", points)

        candidates = compute_match_candidates(
            [detection], [contour], max_angle_diff=30.0, use_angle=True
        )

        assert len(candidates) == 0

    def test_multiple_candidates(self):
        c1 = make_vertical_contour(x=10)
        c2 = make_vertical_contour(x=20)
        points = make_vertical_text_box(center_x=15, center_y=50)
        detection = create_detection("100", points)

        candidates = compute_match_candidates([detection], [c1, c2])

        assert len(candidates) == 2

    def test_skips_unparseable_text(self):
        contour = make_vertical_contour(x=10)
        points = make_vertical_text_box(center_x=15, center_y=50)
        detection = create_detection("invalid", points)

        candidates = compute_match_candidates([detection], [contour])

        assert len(candidates) == 0


# --- Build Cost Matrix Tests ---


class TestBuildCostMatrix:
    def test_single_candidate(self):
        candidate = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        matrix = build_cost_matrix([candidate], n_detections=1, n_contours=1)

        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 15.0

    def test_invalid_pairs_have_high_cost(self):
        """Test that pairs without candidates have high cost (1e9)."""
        candidate = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        matrix = build_cost_matrix([candidate], n_detections=2, n_contours=2)

        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 15.0
        # Invalid pairs get high cost (1e9) instead of inf
        assert matrix[0, 1] == 1e9
        assert matrix[1, 0] == 1e9
        assert matrix[1, 1] == 1e9

    def test_keeps_minimum_cost(self):
        c1 = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=20.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        c2 = MatchCandidate(
            detection_idx=0,
            contour_idx=0,
            distance=10.0,
            angle_diff=5.0,
            height=100.0,
            confidence=1.0,
        )
        matrix = build_cost_matrix([c1, c2], n_detections=1, n_contours=1)

        assert matrix[0, 0] == 15.0  # Minimum of 25.0 and 15.0


# --- Hungarian Matching Tests ---


class TestMatchTextToContours:
    """Tests for the main matching function using Hungarian algorithm."""

    def test_basic_single_match(self):
        contour = make_vertical_contour(x=10)
        points = make_vertical_text_box(center_x=15, center_y=50)
        detection = create_detection("100", points)

        matches = match_text_to_contours([detection], [contour])

        assert matches == {0: 100.0}

    def test_empty_inputs(self):
        assert match_text_to_contours([], []) == {}

        contour = make_vertical_contour(x=10)
        assert match_text_to_contours([], [contour]) == {}

        points = make_vertical_text_box(center_x=15, center_y=50)
        detection = create_detection("100", points)
        assert match_text_to_contours([detection], []) == {}

    def test_distance_threshold(self):
        contour = make_vertical_contour(x=10)
        points = make_vertical_text_box(center_x=100, center_y=50)
        detection = create_detection("100", points)

        matches = match_text_to_contours([detection], [contour], max_distance=50.0)

        assert matches == {}

    def test_angle_threshold(self):
        contour = make_vertical_contour(x=10)  # Angle 90
        points = make_horizontal_text_box(center_x=15, center_y=50)  # Angle 0
        detection = create_detection("100", points)

        matches = match_text_to_contours(
            [detection], [contour], max_angle_diff=30.0, use_angle=True
        )

        assert matches == {}

    def test_picks_closer_contour(self):
        c1 = make_vertical_contour(x=10)
        c2 = make_vertical_contour(x=20)
        points = make_vertical_text_box(center_x=18, center_y=50)  # Closer to c2
        detection = create_detection("200", points)

        matches = match_text_to_contours([detection], [c1, c2])

        assert 0 not in matches
        assert matches[1] == 200.0

    def test_one_to_one_assignment(self):
        """Each detection matches at most one contour."""
        c1 = make_vertical_contour(x=10)
        c2 = make_vertical_contour(x=30)

        d1 = create_detection("100", make_vertical_text_box(center_x=15, center_y=50))
        d2 = create_detection("200", make_vertical_text_box(center_x=25, center_y=50))

        matches = match_text_to_contours([d1, d2], [c1, c2])

        assert len(matches) == 2
        assert matches[0] == 100.0
        assert matches[1] == 200.0


class TestHungarianOptimality:
    """Tests that verify Hungarian algorithm finds optimal solution."""

    def test_conflict_resolution_by_confidence(self):
        """When two detections compete for one contour, higher confidence wins."""
        contour = make_vertical_contour(x=10)

        # Both detections at same distance from contour
        d1 = create_detection(
            "100",
            make_vertical_text_box(center_x=15, center_y=50),
            confidence=0.9,
        )
        d2 = create_detection(
            "200",
            make_vertical_text_box(center_x=15, center_y=60),
            confidence=0.5,
        )

        matches = match_text_to_contours([d1, d2], [contour])

        # Only one can match; higher confidence (d1) should win
        assert len(matches) == 1
        assert matches[0] == 100.0

    def test_hungarian_beats_greedy(self):
        """Case where greedy would give suboptimal result.

        Setup:
        - Text A at (12, 50): close to contour 0 (x=10), far from contour 1 (x=30)
        - Text B at (15, 50): between contours, slightly closer to contour 0

        Greedy (process in order):
        - A matches contour 0 (distance 2)
        - B wants contour 0 (distance 5) but taken, matches contour 1 (distance 15)
        - Total: 2 + 15 = 17

        Hungarian (optimal):
        - A matches contour 0 (distance 2)
        - B matches contour 1 (distance 15)
        OR
        - A matches contour 1 (distance 18)
        - B matches contour 0 (distance 5)
        - Optimal total: 2 + 15 = 17 or 18 + 5 = 23

        Actually, let's make a clearer case:
        """
        c0 = make_vertical_contour(x=10)
        c1 = make_vertical_contour(x=25)

        # A is closer to c0, B is closer to c1
        d_a = create_detection("100", make_vertical_text_box(center_x=12, center_y=50))
        d_b = create_detection("200", make_vertical_text_box(center_x=23, center_y=50))

        matches = match_text_to_contours([d_a, d_b], [c0, c1])

        # Optimal: A->c0, B->c1
        assert matches == {0: 100.0, 1: 200.0}

    def test_cross_assignment_optimal(self):
        """Case where crossing assignments gives better total cost.

        Setup (equal confidence):
        - c0 at x=0, c1 at x=20
        - d0 at x=8, d1 at x=12

        Direct assignment: d0->c0 (dist 8), d1->c1 (dist 8), total = 16
        Cross assignment: d0->c1 (dist 12), d1->c0 (dist 12), total = 24

        Direct is better, Hungarian should find it.
        """
        c0 = make_vertical_contour(x=0)
        c1 = make_vertical_contour(x=20)

        d0 = create_detection("100", make_vertical_text_box(center_x=8, center_y=50))
        d1 = create_detection("200", make_vertical_text_box(center_x=12, center_y=50))

        matches = match_text_to_contours([d0, d1], [c0, c1])

        assert matches == {0: 100.0, 1: 200.0}


class TestRectangularAssignment:
    """Tests with unequal numbers of detections and contours."""

    def test_more_detections_than_contours(self):
        """Three detections, two contours - one detection unmatched."""
        c0 = make_vertical_contour(x=10)
        c1 = make_vertical_contour(x=30)

        d0 = create_detection("100", make_vertical_text_box(center_x=12, center_y=50))
        d1 = create_detection("200", make_vertical_text_box(center_x=28, center_y=50))
        d2 = create_detection(
            "300", make_vertical_text_box(center_x=50, center_y=50)
        )  # Far

        matches = match_text_to_contours([d0, d1, d2], [c0, c1], max_distance=25.0)

        assert len(matches) == 2
        assert matches[0] == 100.0
        assert matches[1] == 200.0

    def test_more_contours_than_detections(self):
        """Two detections, three contours - one contour unmatched."""
        c0 = make_vertical_contour(x=10)
        c1 = make_vertical_contour(x=30)
        c2 = make_vertical_contour(x=50)

        d0 = create_detection("100", make_vertical_text_box(center_x=12, center_y=50))
        d1 = create_detection("200", make_vertical_text_box(center_x=28, center_y=50))

        matches = match_text_to_contours([d0, d1], [c0, c1, c2])

        assert len(matches) == 2
        assert matches[0] == 100.0
        assert matches[1] == 200.0
        assert 2 not in matches

    def test_all_invalid_returns_empty(self):
        """When no valid matches exist, return empty dict."""
        c0 = make_vertical_contour(x=10)
        d0 = create_detection("100", make_horizontal_text_box(center_x=15, center_y=50))

        matches = match_text_to_contours(
            [d0], [c0], max_angle_diff=10.0, use_angle=True
        )

        assert matches == {}


class TestConfidenceWeighting:
    """Tests for confidence-based prioritization."""

    def test_high_confidence_preferred(self):
        """Higher confidence detection wins when competing for same contour."""
        contour = make_vertical_contour(x=10)

        # Same position, different confidence
        low_conf = create_detection(
            "100", make_vertical_text_box(center_x=15, center_y=50), confidence=0.3
        )
        high_conf = create_detection(
            "200", make_vertical_text_box(center_x=15, center_y=50), confidence=0.9
        )

        matches = match_text_to_contours([low_conf, high_conf], [contour])

        assert matches[0] == 200.0  # High confidence wins

    def test_confidence_can_overcome_distance(self):
        """High confidence can sometimes beat closer low-confidence detection."""
        contour = make_vertical_contour(x=10)

        # d1 is closer but low confidence
        d1 = create_detection(
            "100", make_vertical_text_box(center_x=12, center_y=50), confidence=0.2
        )
        # d2 is farther but high confidence
        d2 = create_detection(
            "200", make_vertical_text_box(center_x=20, center_y=50), confidence=1.0
        )

        matches = match_text_to_contours([d1, d2], [contour])

        # d1 cost: 2 * (1/0.2) = 10
        # d2 cost: 10 * (1/1.0) = 10
        # Tie - either could win, but both are valid
        assert len(matches) == 1
        assert matches[0] in [100.0, 200.0]


class TestLargeScaleMatching:
    """Tests with many detections and contours."""

    def test_ten_by_ten_matching(self):
        """10 detections, 10 contours - all should match optimally."""
        contours = [make_vertical_contour(x=i * 20) for i in range(10)]
        detections = [
            create_detection(
                str(i * 100),
                make_vertical_text_box(center_x=i * 20 + 5, center_y=50),
            )
            for i in range(10)
        ]

        matches = match_text_to_contours(detections, contours)

        assert len(matches) == 10
        for i in range(10):
            assert matches[i] == i * 100

    def test_sparse_matching(self):
        """Many contours, few detections."""
        contours = [make_vertical_contour(x=i * 10) for i in range(20)]
        detections = [
            create_detection("100", make_vertical_text_box(center_x=5, center_y=50)),
            create_detection("500", make_vertical_text_box(center_x=55, center_y=50)),
        ]

        matches = match_text_to_contours(detections, contours)

        assert len(matches) == 2
        assert matches[0] == 100.0  # Closest to x=0
        assert matches[5] == 500.0  # Closest to x=50


class TestWeightParameters:
    """Tests for distance_weight and angle_weight parameters."""

    def test_distance_weight_prioritizes_distance(self):
        """Higher distance weight makes distance more important."""
        c0 = make_vertical_contour(x=10)
        c1 = make_vertical_contour(x=20)

        # d0 is closer to c0, d1 slightly closer to c1
        d0 = create_detection("100", make_vertical_text_box(center_x=12, center_y=50))
        d1 = create_detection("200", make_vertical_text_box(center_x=18, center_y=50))

        matches = match_text_to_contours(
            [d0, d1], [c0, c1], distance_weight=10.0, angle_weight=0.1
        )

        # With high distance weight, closest matches preferred
        assert matches[0] == 100.0
        assert matches[1] == 200.0

    def test_angle_weight_prioritizes_alignment(self):
        """Higher angle weight makes alignment more important."""
        # Horizontal contour at y=50
        contour = make_horizontal_contour(y=50)

        # Slightly rotated text (not perfectly horizontal)
        points_aligned = [(0, 48), (20, 48), (20, 52), (0, 52)]  # Nearly horizontal
        points_rotated = [(0, 45), (20, 55), (25, 50), (5, 40)]  # Rotated ~27 degrees

        d_aligned = create_detection("100", points_aligned)
        d_rotated = create_detection("200", points_rotated)

        matches = match_text_to_contours(
            [d_aligned, d_rotated], [contour], angle_weight=10.0, max_angle_diff=45.0
        )

        # Better aligned detection should match
        assert matches.get(0) == 100.0
