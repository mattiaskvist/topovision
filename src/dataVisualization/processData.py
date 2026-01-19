"""Usage in main below! It processes one file, though can download large data.

Contour Map Dataset Generator with Dynamic Splitting
=====================================================
Input: .shp/.geojson with contour lines.and 'elevation' column
Output: Multiple PNG + JSON tiles with no overlapping labels

Key features:
- Smart adaptive splitting based on label collisions
- all labels included - recursively splits until no collisions
- Highly varied label styling (colors, fonts, sizes, rotations)

JSON format per tile:
{
  "image": {"path": "<tile>.png", "size": {"height": H, "width": W}},
  "labels": [
    {"elevation": <int>, "elevation_bbox_pixels": [x,y,w,h],
     "contour_line_pixels": [[x,y],...]}
  ]
}
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import colorsys
import json
import random
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
from shapely.ops import clip_by_rect

ELEV_COLUMN = "elevation"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class LabelCandidate:
    """A potential label to be placed."""

    elevation: int
    anchor_x: float  # World coordinates
    anchor_y: float
    line_coords: list  # Original contour line coordinates
    geom_idx: int  # Index in the geodataframe


@dataclass
class PlacedLabel:
    """A label that has been successfully placed."""

    elevation: int
    anchor_x: float
    anchor_y: float
    bbox_pixels: list  # [x, y, w, h] in pixels
    contour_line_pixels: list  # [[x, y], ...] in pixels


# =============================================================================
# COLOR GENERATION WITH CONTRAST
# =============================================================================


def generate_contrasting_colors(rng: random.Random) -> tuple[str, str, float]:
    """Generate random text and background colors with guaranteed contrast."""
    hue = rng.random()

    style = rng.choice(["light_bg", "dark_bg", "colored_bg", "inverted", "vibrant"])

    if style == "light_bg":
        bg_lightness = rng.uniform(0.85, 0.98)
        text_lightness = rng.uniform(0.0, 0.25)
        bg_sat = rng.uniform(0.0, 0.3)
        text_sat = rng.uniform(0.0, 0.4)
    elif style == "dark_bg":
        bg_lightness = rng.uniform(0.05, 0.25)
        text_lightness = rng.uniform(0.8, 0.98)
        bg_sat = rng.uniform(0.0, 0.5)
        text_sat = rng.uniform(0.0, 0.3)
    elif style == "colored_bg":
        bg_lightness = rng.uniform(0.4, 0.7)
        bg_sat = rng.uniform(0.3, 0.8)
        text_lightness = (
            rng.uniform(0.0, 0.15) if rng.random() < 0.5 else rng.uniform(0.9, 1.0)
        )
        text_sat = rng.uniform(0.0, 0.2)
    elif style == "inverted":
        bg_lightness = rng.uniform(0.1, 0.3)
        text_lightness = rng.uniform(0.85, 1.0)
        bg_sat = rng.uniform(0.5, 0.9)
        text_sat = rng.uniform(0.0, 0.2)
        hue = (hue + 0.5) % 1.0
    else:  # vibrant
        bg_lightness = rng.uniform(0.5, 0.75)
        bg_sat = rng.uniform(0.6, 1.0)
        text_lightness = rng.uniform(0.0, 0.1)
        text_sat = 0.0

    bg_rgb = colorsys.hls_to_rgb(hue, bg_lightness, bg_sat)
    text_hue = hue if rng.random() < 0.7 else rng.random()
    text_rgb = colorsys.hls_to_rgb(text_hue, text_lightness, text_sat)

    bg_color = (
        f"#{int(bg_rgb[0] * 255):02x}"
        f"{int(bg_rgb[1] * 255):02x}"
        f"{int(bg_rgb[2] * 255):02x}"
    )
    text_color = (
        f"#{int(text_rgb[0] * 255):02x}"
        f"{int(text_rgb[1] * 255):02x}"
        f"{int(text_rgb[2] * 255):02x}"
    )

    return text_color, bg_color, rng.uniform(0.6, 0.95)


def random_colormap(seed: int) -> mcolors.LinearSegmentedColormap:
    """Generate a random terrain-style colormap."""
    rng = np.random.RandomState(seed)

    scheme_type = rng.choice(
        [
            "natural",
            "vibrant",
            "monochrome",
            "desert",
            "ocean",
            "volcanic",
            "alien",
            "pastel",
            "neon",
        ]
    )

    schemes = {
        "natural": [
            "#3a7ca5",
            "#5fa370",
            "#8fb339",
            "#c8b078",
            "#a0896a",
            "#7a6855",
            "#a0a0a0",
            "#d0d0d0",
            "#f0f0f0",
        ],
        "vibrant": [
            "#0077be",
            "#00c853",
            "#ffd600",
            "#ff6f00",
            "#d84315",
            "#8e24aa",
            "#5e35b1",
            "#1e88e5",
            "#e0e0e0",
        ],
        "desert": [
            "#8b7355",
            "#a08060",
            "#b89070",
            "#c8a080",
            "#d8b898",
            "#e0c8a8",
            "#e8d8c0",
            "#f0e8d8",
            "#f8f0e8",
        ],
        "ocean": [
            "#001a33",
            "#003366",
            "#004d99",
            "#0066cc",
            "#0080ff",
            "#33adff",
            "#66c2ff",
            "#99d6ff",
            "#ccebff",
        ],
        "volcanic": [
            "#1a0f0a",
            "#331a0f",
            "#4d2614",
            "#663319",
            "#804020",
            "#995533",
            "#b36b4d",
            "#cc8866",
            "#e6a380",
        ],
        "pastel": [
            "#ffd6e0",
            "#ffe0cc",
            "#fff0cc",
            "#ffffcc",
            "#e6ffcc",
            "#ccffe6",
            "#ccf0ff",
            "#e0ccff",
            "#ffccf0",
        ],
        "neon": [
            "#0a0a0a",
            "#1a0a2e",
            "#16213e",
            "#0f3460",
            "#e94560",
            "#ff6b6b",
            "#ffd93d",
            "#6bcb77",
            "#4d96ff",
        ],
    }

    if scheme_type == "alien":
        base = [
            "#" + "".join(rng.choice(list("0123456789abcdef")) for _ in range(6))
            for _ in range(9)
        ]
    elif scheme_type == "monochrome":
        tint = rng.choice(["blues", "greens", "reds", "purples"])
        tint_colors = {
            "blues": ["#1a2332", "#2d3e50", "#4a5f7f", "#6a7fa0", "#8fa0bf", "#b0c0d0"],
            "greens": [
                "#1a2618",
                "#2d3e2a",
                "#4a5f44",
                "#6a7f60",
                "#8fa084",
                "#b0c0a8",
            ],
            "reds": ["#321a1a", "#502d2d", "#7f4a4a", "#a06a6a", "#bf8f8f", "#d0b0b0"],
            "purples": [
                "#2a1a32",
                "#3e2d50",
                "#5f4a7f",
                "#7f6aa0",
                "#a08fbf",
                "#c0b0d0",
            ],
        }
        base = tint_colors[tint]
    else:
        base = schemes.get(scheme_type, schemes["natural"])

    colors = []
    for color in base:
        if rng.random() < 0.3:
            rgb = mcolors.hex2color(color)
            rgb = tuple(np.clip(c + rng.uniform(-0.15, 0.15), 0, 1) for c in rgb)
            colors.append(rgb)
        else:
            colors.append(color)

    if rng.random() < 0.2:
        colors.reverse()

    return mcolors.LinearSegmentedColormap.from_list("terrain", colors)


# =============================================================================
# COORDINATE TRANSFORMS
# =============================================================================


def to_pixel(x: float, y: float, bounds: tuple, size: int) -> tuple[int, int]:
    """Convert world coordinates to pixel coordinates."""
    minx, miny, maxx, maxy = bounds
    if maxx <= minx or maxy <= miny:
        return 0, 0
    px = (x - minx) / (maxx - minx) * size
    py = (maxy - y) / (maxy - miny) * size
    return round(px), round(py)


def estimate_label_bbox_pixels(
    anchor_x: float,
    anchor_y: float,
    elevation: int,
    bounds: tuple,
    size: int,
    fontsize: float = 9,
) -> list:
    """Estimate label bounding box in pixels without rendering.

    This is an approximation used for fast collision detection.
    """
    px, py = to_pixel(anchor_x, anchor_y, bounds, size)

    # Estimate text dimensions based on fontsize and number of digits
    num_digits = len(str(abs(elevation)))
    char_width = fontsize * 0.7
    char_height = fontsize * 1.2

    # Add padding for bbox
    padding = fontsize * 0.4

    w = num_digits * char_width + padding * 2
    h = char_height + padding * 2

    # Center the box
    x = px - w / 2
    y = py - h / 2

    return [x, y, w, h]


def boxes_overlap(box1: list, box2: list, margin: int = 5) -> bool:
    """Check if two bounding boxes overlap."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (
        x1 + w1 + margin < x2
        or x2 + w2 + margin < x1
        or y1 + h1 + margin < y2
        or y2 + h2 + margin < y1
    )


# =============================================================================
# COLLISION-BASED SPLITTING (FAST, NO RENDERING)
# =============================================================================


def collect_label_candidates(
    gdf: gpd.GeoDataFrame, bounds: tuple, seed: int
) -> list[LabelCandidate]:
    """Collect all potential labels from geometries."""
    rng = random.Random(seed)
    candidates = []

    for idx, row in gdf.iterrows():
        if ELEV_COLUMN not in row or row.geometry is None or row.geometry.is_empty:
            continue

        elev = int(row[ELEV_COLUMN])
        line = row.geometry

        # Get point on line (random position between 20-80%)
        fraction = rng.uniform(0.2, 0.8)
        try:
            anchor = line.interpolate(fraction * line.length)
        except Exception:
            continue

        try:
            line_coords = list(line.coords)
        except Exception:
            line_coords = []

        candidates.append(
            LabelCandidate(
                elevation=elev,
                anchor_x=anchor.x,
                anchor_y=anchor.y,
                line_coords=line_coords,
                geom_idx=idx,
            )
        )

    return candidates


def find_collisions_fast(
    candidates: list[LabelCandidate], bounds: tuple, size: int
) -> tuple[list[int], list[int]]:
    """Find which labels collide using estimated bboxes.

    Returns (placed_indices, collision_indices)
    """
    placed = []
    placed_boxes = []
    collisions = []

    for i, cand in enumerate(candidates):
        box = estimate_label_bbox_pixels(
            cand.anchor_x, cand.anchor_y, cand.elevation, bounds, size
        )

        has_collision = False
        for placed_box in placed_boxes:
            if boxes_overlap(box, placed_box):
                has_collision = True
                break

        if has_collision:
            collisions.append(i)
        else:
            placed.append(i)
            placed_boxes.append(box)

    return placed, collisions


def compute_split_bounds(
    bounds: tuple, collision_candidates: list[LabelCandidate], size: int
) -> list[tuple]:
    """Compute smart split based on collision locations.

    Returns list of sub-bounds, filtering out any that are too small.
    """
    minx, miny, maxx, maxy = bounds
    range_x = maxx - minx
    range_y = maxy - miny

    # Minimum size threshold (avoid thin slices)
    # Don't create tiles smaller than 15% of original in either dimension
    min_fraction = 0.15

    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2

    if collision_candidates:
        # Find average position of collisions in world coords
        coll_x = [c.anchor_x for c in collision_candidates]
        coll_y = [c.anchor_y for c in collision_candidates]

        # Use median for split point, but clamp to reasonable range
        split_x = np.median(coll_x)
        split_y = np.median(coll_y)

        # Clamp to 35-65% range to produce more balanced splits
        split_x = max(minx + 0.35 * range_x, min(maxx - 0.35 * range_x, split_x))
        split_y = max(miny + 0.35 * range_y, min(maxy - 0.35 * range_y, split_y))
    else:
        split_x, split_y = midx, midy

    # Generate candidate bounds
    all_bounds = [
        (minx, miny, split_x, split_y),
        (split_x, miny, maxx, split_y),
        (minx, split_y, split_x, maxy),
        (split_x, split_y, maxx, maxy),
    ]

    # Filter out bounds that are too small
    valid_bounds = []
    for b in all_bounds:
        bx_range = b[2] - b[0]
        by_range = b[3] - b[1]
        # Only include if both dimensions are at least min_fraction of original
        if bx_range >= range_x * min_fraction and by_range >= range_y * min_fraction:
            valid_bounds.append(b)

    # If all were filtered (shouldn't happen), return original bounds
    return valid_bounds if valid_bounds else all_bounds


def clip_gdf_to_bounds(gdf: gpd.GeoDataFrame, bounds: tuple) -> gpd.GeoDataFrame:
    """Clip geodataframe to bounds."""
    minx, miny, maxx, maxy = bounds

    # Spatial indexing
    clipped = gdf.cx[minx:maxx, miny:maxy].copy()

    if clipped.empty:
        return clipped

    # Clip geometries
    def clip_geom(geom):
        try:
            return clip_by_rect(geom, minx, miny, maxx, maxy)
        except Exception:
            return geom

    clipped["geometry"] = clipped.geometry.apply(clip_geom)
    clipped = clipped[~clipped.geometry.is_empty]

    return clipped


def find_non_colliding_regions(
    gdf: gpd.GeoDataFrame,
    bounds: tuple,
    seed: int,
    size: int,
    depth: int = 0,
    max_depth: int = 200,
) -> list[tuple[gpd.GeoDataFrame, tuple, int]]:
    """Recursively find regions where labels don't collide.

    Returns list of (gdf, bounds, seed) tuples for regions ready to render.
    """
    clipped = clip_gdf_to_bounds(gdf, bounds)

    if clipped.empty:
        return []

    # Collect label candidates
    candidates = collect_label_candidates(clipped, bounds, seed)

    if not candidates:
        return []

    # Fast collision check
    _placed_indices, collision_indices = find_collisions_fast(candidates, bounds, size)

    # If no collisions, this region is ready
    if not collision_indices or depth >= max_depth:
        return [(clipped, bounds, seed)]

    # Need to split - compute split bounds based on collisions
    collision_candidates = [candidates[i] for i in collision_indices]
    sub_bounds_list = compute_split_bounds(bounds, collision_candidates, size)

    results = []
    for i, sub_bounds in enumerate(sub_bounds_list):
        sub_seed = seed + i + depth * 4
        sub_results = find_non_colliding_regions(
            gdf, sub_bounds, sub_seed, size, depth + 1, max_depth
        )
        results.extend(sub_results)

    return results


# =============================================================================
# FINAL RENDERING
# =============================================================================


def generate_label_style(rng: random.Random) -> dict:
    """Generate random style for a single label.

    Note: This function generates diverse label styles to create varied training data.
    30% of labels have no background (matching real maps), while 70% have contrasting
    backgrounds to help the model learn to read text in various conditions.
    """
    text_color, bg_color, alpha = generate_contrasting_colors(rng)

    fontsize = rng.uniform(6, 14)
    weight = rng.choice(["normal", "bold", "bold", "bold"])
    family = rng.choice(
        ["sans-serif", "serif", "monospace", "sans-serif", "sans-serif"]
    )
    rotation = rng.uniform(-12, 12)
    padding = rng.uniform(0.15, 0.45)

    boxstyle = rng.choice(
        [
            f"round,pad={padding}",
            f"round,pad={padding},rounding_size=0.3",
            f"square,pad={padding}",
            f"roundtooth,pad={padding}",
            f"sawtooth,pad={padding}",
        ]
    )

    edge_color = text_color if rng.random() < 0.2 else "none"
    edge_width = rng.uniform(0.5, 1.5) if edge_color != "none" else 0

    # 30% of labels have no background box (raw text on contour lines)
    if rng.random() < 0.3:
        bbox_style = None
    else:
        bbox_style = dict(
            boxstyle=boxstyle,
            facecolor=bg_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            alpha=alpha,
        )

    return {
        "fontsize": fontsize,
        "weight": weight,
        "family": family,
        "color": text_color,
        "rotation": rotation,
        "bbox": bbox_style,
    }


def add_secondary_color_pattern(ax, bounds: tuple, grid_res: int, seed: int, rng):
    """Add secondary color overlay pattern for visual variety."""
    minx, miny, maxx, maxy = bounds

    # Different colormap for variety
    overlay_cmap = random_colormap(seed + 777)

    # Different noise pattern
    overlay_pattern = gaussian_filter(
        rng.randn(grid_res, grid_res), sigma=rng.uniform(8, 15)
    )
    overlay_pattern = overlay_pattern - overlay_pattern.min()
    if overlay_pattern.max() > 0:
        overlay_pattern = overlay_pattern / overlay_pattern.max()

    ax.imshow(
        overlay_pattern,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap=overlay_cmap,
        aspect="auto",
        interpolation="bilinear",
        alpha=rng.uniform(0.15, 0.35),
        zorder=0.05,
    )


def add_feature_patches(ax, bounds: tuple, grid_res: int, rng):
    """Add distinct color patches for terrain features."""
    minx, miny, maxx, maxy = bounds

    # Create blob-like regions
    blob_noise = gaussian_filter(
        rng.randn(grid_res, grid_res), sigma=rng.uniform(10, 18)
    )
    threshold = rng.uniform(-0.3, 0.3)
    blob_mask = (blob_noise > threshold).astype(float)
    blob_mask = gaussian_filter(blob_mask, sigma=2)  # Soft edges

    # Pick a distinct color for the patch
    patch_colors = [
        "#2d5a27",
        "#1e4d6b",
        "#6b4423",
        "#4a3b5c",
        "#5c6b3b",
        "#6b3b5c",
        "#3b5c6b",
        "#6b5c3b",
        "#3b6b5c",
        "#5c3b6b",
    ]
    patch_color = rng.choice(patch_colors)

    # Create colored patch
    patch_rgb = mcolors.to_rgb(patch_color)
    patch_img = np.zeros((grid_res, grid_res, 4))
    patch_img[:, :, 0] = patch_rgb[0]
    patch_img[:, :, 1] = patch_rgb[1]
    patch_img[:, :, 2] = patch_rgb[2]
    patch_img[:, :, 3] = blob_mask * rng.uniform(0.2, 0.45)

    ax.imshow(
        patch_img,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        zorder=0.08,
    )


def add_brightness_variation(ax, bounds: tuple, grid_res: int, rng):
    """Add brightness variation with radial gradients."""
    minx, miny, maxx, maxy = bounds

    # Create radial gradient
    xx, yy = np.meshgrid(np.linspace(-1, 1, grid_res), np.linspace(-1, 1, grid_res))
    # Random center offset
    cx, cy = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = 1 - np.clip(radial / 1.5, 0, 1)

    # Apply as brightness
    if rng.random() < 0.5:
        # Dark edges (vignette)
        ax.imshow(
            radial,
            extent=[minx, maxx, miny, maxy],
            origin="lower",
            cmap="gray",
            aspect="auto",
            alpha=rng.uniform(0.1, 0.25),
            zorder=0.02,
        )
    else:
        # Light center (spotlight)
        ax.imshow(
            1 - radial,
            extent=[minx, maxx, miny, maxy],
            origin="lower",
            cmap="gray_r",
            aspect="auto",
            alpha=rng.uniform(0.05, 0.15),
            zorder=0.02,
        )


def render_terrain_background(ax, gdf, elev_col: str, bounds: tuple, seed: int):
    """Render terrain background with RICH color variation across entire tile.

    Creates terrain-like patterns with multiple color zones (like lakes, forests,
    mountains etc) regardless of the actual elevation range in this tile.
    """
    minx, miny, maxx, maxy = bounds
    elevations = gdf[elev_col].values
    elev_min, elev_max = elevations.min(), elevations.max()

    rng = np.random.RandomState(seed)

    # Grid resolution
    grid_res = rng.choice([80, 100, 120])

    # === CREATE RICH SPATIAL VARIATION ===

    # Multiple noise layers at different scales create terrain-like patterns
    # These represent different "zones" like water, forest, desert, mountains etc

    # Base layer: large terrain features
    base_noise = gaussian_filter(
        rng.randn(grid_res, grid_res), sigma=rng.uniform(12, 20)
    )

    # Secondary features
    secondary_noise = gaussian_filter(
        rng.randn(grid_res, grid_res), sigma=rng.uniform(5, 10)
    )

    # Tertiary features
    tertiary_noise = gaussian_filter(
        rng.randn(grid_res, grid_res), sigma=rng.uniform(2, 5)
    )

    # Fine detail
    detail_noise = gaussian_filter(
        rng.randn(grid_res, grid_res), sigma=rng.uniform(0.5, 2)
    )

    # Combine with random weights
    w1 = rng.uniform(0.3, 0.5)
    w2 = rng.uniform(0.2, 0.35)
    w3 = rng.uniform(0.1, 0.25)
    w4 = rng.uniform(0.05, 0.15)

    # Normalize weights
    total_w = w1 + w2 + w3 + w4
    terrain_pattern = (
        base_noise * w1 + secondary_noise * w2 + tertiary_noise * w3 + detail_noise * w4
    ) / total_w

    # Normalize to 0-1 range with good spread
    terrain_pattern = terrain_pattern - terrain_pattern.min()
    if terrain_pattern.max() > 0:
        terrain_pattern = terrain_pattern / terrain_pattern.max()

    # Add some contrast enhancement
    contrast = rng.uniform(0.8, 1.4)
    terrain_pattern = np.clip((terrain_pattern - 0.5) * contrast + 0.5, 0, 1)

    # === OPTIONAL: Mix in actual contour elevation data ===
    if elev_max > elev_min and rng.random() < 0.5:
        # Sometimes blend in real elevation for more variety
        x_grid = np.linspace(minx, maxx, grid_res)
        y_grid = np.linspace(miny, maxy, grid_res)
        z_grid = np.ones((grid_res, grid_res)) * elev_min

        buffer_size = (maxx - minx) / grid_res * rng.uniform(2.0, 4.0)
        for elev in sorted(gdf[elev_col].unique()):
            for geom in gdf[gdf[elev_col] == elev].geometry:
                try:
                    buffered = geom.buffer(buffer_size)
                    gminx, gminy, gmaxx, gmaxy = buffered.bounds

                    i_start = max(0, int((gminx - minx) / (maxx - minx) * grid_res))
                    i_end = min(
                        grid_res, int((gmaxx - minx) / (maxx - minx) * grid_res) + 1
                    )
                    j_start = max(0, int((gminy - miny) / (maxy - miny) * grid_res))
                    j_end = min(
                        grid_res, int((gmaxy - miny) / (maxy - miny) * grid_res) + 1
                    )

                    step = max(1, (i_end - i_start) // 12)
                    for i in range(i_start, i_end, step):
                        for j in range(j_start, j_end, step):
                            if (
                                i < grid_res
                                and j < grid_res
                                and buffered.contains(Point(x_grid[i], y_grid[j]))
                            ):
                                z_grid[j, i] = elev
                except Exception:
                    continue

        z_grid = gaussian_filter(z_grid, sigma=1.5)
        z_norm = (z_grid - elev_min) / (elev_max - elev_min)

        # Blend elevation with spatial pattern
        blend = rng.uniform(0.2, 0.5)
        terrain_pattern = terrain_pattern * (1 - blend) + z_norm * blend

    # === COLORMAP AND RENDERING ===

    terrain_cmap = random_colormap(seed)

    # Random interpolation
    interp = rng.choice(["bilinear", "bicubic", "lanczos"])
    alpha = rng.uniform(0.75, 0.98)

    # Main terrain layer
    ax.imshow(
        terrain_pattern,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap=terrain_cmap,
        aspect="auto",
        interpolation=interp,
        alpha=alpha,
        zorder=0,
    )

    # === ADD SECONDARY COLOR PATTERN ===

    if rng.random() < 0.6:  # 60% chance for extra color layer
        add_secondary_color_pattern(ax, bounds, grid_res, seed, rng)

    # === OCCASIONAL FEATURE PATCHES ===

    if rng.random() < 0.4:  # 40% chance for distinct patches
        add_feature_patches(ax, bounds, grid_res, rng)

    # Brightness variation
    if rng.random() < 0.2:
        add_brightness_variation(ax, bounds, grid_res, rng)

    return elev_min, elev_max, terrain_cmap


def render_final_tile(
    gdf: gpd.GeoDataFrame,
    bounds: tuple,
    img_path: Path,
    seed: int,
    size: int = 512,
    dpi: int = 150,
) -> list[dict]:
    """Render a tile that has been verified to have no collisions."""
    if gdf.empty or ELEV_COLUMN not in gdf.columns:
        return []

    rng = random.Random(seed)
    minx, miny, maxx, maxy = bounds

    # Create figure
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")

    # Render terrain
    elev_min, elev_max, terrain_cmap = render_terrain_background(
        ax, gdf, ELEV_COLUMN, bounds, seed
    )

    # Draw contour lines
    line_width = rng.uniform(0.8, 1.5)
    for _, row in gdf.iterrows():
        elev = row[ELEV_COLUMN]
        norm_elev = (
            (elev - elev_min) / (elev_max - elev_min) if elev_max > elev_min else 0.5
        )

        if terrain_cmap:
            color = terrain_cmap(norm_elev)
            darken_factor = rng.uniform(0.3, 0.6)
            line_color = (*tuple(c * darken_factor for c in color[:3]), 1.0)
        else:
            line_color = "black"

        gdf[gdf[ELEV_COLUMN] == elev].plot(
            ax=ax,
            linewidth=line_width,
            edgecolor=line_color,
            facecolor="none",
            zorder=2,
        )

    # Add labels
    text_objects = []

    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        elev = int(row[ELEV_COLUMN])
        line = row.geometry

        fraction = rng.uniform(0.2, 0.8)
        try:
            anchor = line.interpolate(fraction * line.length)
        except Exception:
            continue

        style = generate_label_style(rng)

        txt = ax.text(
            anchor.x,
            anchor.y,
            str(elev),
            fontsize=style["fontsize"],
            weight=style["weight"],
            family=style["family"],
            ha="center",
            va="center",
            color=style["color"],
            rotation=style["rotation"],
            bbox=style["bbox"],
            zorder=3,
        )

        try:
            line_coords = list(line.coords)
        except Exception:
            line_coords = []

        text_objects.append((txt, elev, line_coords))

    # Get actual bboxes and do final collision check
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    transform = fig.transFigure.inverted()

    placed_boxes = []
    labels = []

    for txt, elev, line_coords in text_objects:
        try:
            bbox_disp = txt.get_window_extent(renderer)
            bbox_fig = bbox_disp.transformed(transform)

            x = bbox_fig.x0 * size
            y = (1 - bbox_fig.y1) * size
            w = bbox_fig.width * size
            h = bbox_fig.height * size
            box = [x, y, w, h]

            if not any(boxes_overlap(box, pb) for pb in placed_boxes):
                contour_pixels = [
                    to_pixel(px, py, bounds, size) for px, py in line_coords
                ]
                labels.append(
                    {
                        "elevation": elev,
                        "elevation_bbox_pixels": box,
                        "contour_line_pixels": contour_pixels,
                    }
                )
                placed_boxes.append(box)
            else:
                txt.set_visible(False)
        except Exception:
            txt.set_visible(False)

    # Save
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)

    return labels


# =============================================================================
# PROCESSING
# =============================================================================


def process_file(input_path: Path, output_dir: Path, size: int = 512, dpi: int = 150):
    """Process one shapefile with adaptive splitting."""
    print(f"📂 Loading {input_path.name}...")
    gdf = gpd.read_file(input_path)

    if ELEV_COLUMN not in gdf.columns:
        err_msg = f"Missing '{ELEV_COLUMN}' column. Found: {list(gdf.columns)}"
        raise ValueError(err_msg)

    elev_range = (gdf[ELEV_COLUMN].min(), gdf[ELEV_COLUMN].max())
    print(f"   Elevation: {elev_range[0]:.0f}m - {elev_range[1]:.0f}m")
    print(f"   Contour lines: {len(gdf)}")

    name = input_path.stem
    tiles_dir = output_dir / name
    tiles_dir.mkdir(parents=True, exist_ok=True)

    bounds = tuple(gdf.total_bounds)
    seed = hash(input_path.name) % (2**31)

    print("   Finding non-colliding regions...")

    # Phase 1: Find all non-colliding regions
    regions = find_non_colliding_regions(gdf, bounds, seed, size)

    print(f"   Found {len(regions)} regions to render")
    print("   Rendering tiles...")

    # Phase 2: Render only the final tiles
    all_tiles = []
    for i, (region_gdf, region_bounds, region_seed) in enumerate(regions):
        tile_id = f"{name}_{i:04d}"
        img_path = tiles_dir / f"{tile_id}.png"
        labels_path = tiles_dir / f"{tile_id}_labels.json"

        labels = render_final_tile(
            region_gdf, region_bounds, img_path, region_seed, size, dpi
        )

        # Save labels JSON
        with open(labels_path, "w") as f:
            json.dump(
                {
                    "image": {
                        "path": f"{tile_id}.png",
                        "size": {"height": size, "width": size},
                    },
                    "labels": labels,
                },
                f,
                indent=2,
            )

        all_tiles.append(
            {
                "tile_id": tile_id,
                "bounds": list(region_bounds),
                "num_labels": len(labels),
            }
        )

        if (i + 1) % 50 == 0:
            print(f"      Rendered {i + 1}/{len(regions)} tiles...")

    if not all_tiles:
        print("No tiles generated")
        return

    # Save summary
    total_labels = sum(t["num_labels"] for t in all_tiles)

    with open(tiles_dir / "summary.json", "w") as f:
        json.dump(
            {
                "source": input_path.name,
                "config": {"tile_size": size, "dpi": dpi, "method": "adaptive_split"},
                "elevation_range": [float(elev_range[0]), float(elev_range[1])],
                "total_tiles": len(all_tiles),
                "total_labels": total_labels,
                "tiles": all_tiles,
            },
            f,
            indent=2,
        )

    print(f"\nComplete: {len(all_tiles)} tiles, {total_labels} labels")
    print(f"Output: {tiles_dir}\n")


def main():
    """Generate contour dataset with adaptive splitting."""
    parser = argparse.ArgumentParser(
        description="Generate contour dataset with adaptive splitting."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/dataVisualization/dataExample/N63E016/N63E016.shp",
        help="Input .shp or .geojson",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/dataVisualization/output/new/example_output",
        help="Output directory",
    )
    parser.add_argument("--size", type=int, default=512, help="Tile size (square)")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_file(input_path, output_dir, size=args.size, dpi=args.dpi)


if __name__ == "__main__":
    main()
