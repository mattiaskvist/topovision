import numpy as np

from height_extraction.inference import (
    build_adjacency_graph,
    build_hierarchy,
    infer_missing_heights,
)


def create_square_contour(x, y, size):
    return np.array(
        [[[x, y]], [[x + size, y]], [[x + size, y + size]], [[x, y + size]]],
        dtype=np.int32,
    )


def test_build_adjacency_graph_empty():
    adj = build_adjacency_graph([], (100, 100))
    assert adj == {}


def test_build_adjacency_graph_disjoint():
    # Two far apart squares
    c1 = create_square_contour(0, 0, 10)
    c2 = create_square_contour(100, 100, 10)
    contours = [c1, c2]

    adj = build_adjacency_graph(contours, (200, 200), max_dist=10.0)
    assert 1 not in adj[0]
    assert 0 not in adj[1]


def test_build_adjacency_graph_adjacent():
    # Two close squares
    c1 = create_square_contour(0, 0, 10)
    c2 = create_square_contour(12, 0, 10)  # 2 pixels gap
    contours = [c1, c2]

    adj = build_adjacency_graph(contours, (100, 100), max_dist=5.0)
    assert 1 in adj[0]
    assert 0 in adj[1]


def test_build_hierarchy_nested():
    # c1 inside c2
    c1 = create_square_contour(40, 40, 20)  # Inner
    c2 = create_square_contour(20, 20, 60)  # Outer
    contours = [c1, c2]

    hierarchy = build_hierarchy(contours)
    assert hierarchy[0] == 1  # c1 parent is c2
    assert hierarchy[1] == -1  # c2 has no parent


def test_infer_missing_heights_single_known():
    # c1 adjacent to c2. c1 known=100. c2 unknown.
    c1 = create_square_contour(0, 0, 10)
    c2 = create_square_contour(12, 0, 10)
    contours = [c1, c2]

    known_heights = {0: 100.0}
    adjacency = {0: {1}, 1: {0}}

    # Should NOT infer c2 because there is no gradient or nesting info.
    # The function returns a dict of inferred heights.
    # If it can't infer, it shouldn't add it to the dict.

    inferred = infer_missing_heights(contours, known_heights, adjacency)

    assert 1 not in inferred
    assert inferred[0] == 100.0


def test_infer_missing_heights_nested_hill():
    # c1 (inner) inside c2 (outer). c2 known=100.
    # Default interval 10.
    # Should assume hill: inner > outer -> c1 = 110.
    c1 = create_square_contour(40, 40, 20)
    c2 = create_square_contour(20, 20, 60)
    contours = [c1, c2]

    known_heights = {1: 100.0}
    adjacency = {0: {1}, 1: {0}}  # They are adjacent spatially too usually

    # We need to mock hierarchy inside the function or trust build_hierarchy works.
    # The function calls build_hierarchy internally.

    inferred = infer_missing_heights(
        contours, known_heights, adjacency, default_contour_interval=10.0
    )

    assert inferred[0] == 110.0


def test_infer_missing_heights_interpolation():
    # c1(100) -- c2(?) -- c3(120)
    c1 = create_square_contour(0, 0, 10)
    c2 = create_square_contour(15, 0, 10)
    c3 = create_square_contour(30, 0, 10)
    contours = [c1, c2, c3]

    known_heights = {0: 100.0, 2: 120.0}
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}}

    inferred = infer_missing_heights(contours, known_heights, adjacency)
    assert inferred[1] == 110.0
