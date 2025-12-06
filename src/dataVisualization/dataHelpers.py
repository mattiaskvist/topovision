"""OCR training dataset generator for contour maps.

Generates image-label pairs for training models to read elevation text from contour maps.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, MultiLineString, Point

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WorldCoordinates:
    """Real-world geographic bounds (WGS84)."""
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2

    def to_dict(self) -> dict:
        lat, lon = self.center
        return {
            "bounds": [self.min_lon, self.min_lat, self.max_lon, self.max_lat],
            "center": [lat, lon],
        }


@dataclass
class TextLabel:
    """Elevation label with bounding box coordinates."""
    elevation: int
    bbox: list[float]
    bbox_normalized: list[float]
    
    def to_dict(self) -> dict:
        return {
            "elevation": self.elevation,
            "bbox_pixels": self.bbox,
            "bbox_normalized": self.bbox_normalized,
        }


@dataclass
class TileMetadata:
    """Tile metadata with world coordinates."""
    tile_id: str
    grid_pos: tuple[int, int]
    world_coords: WorldCoordinates
    image_path: str
    labels_path: str
    elevation_range: tuple[float, float] | None
    num_labels: int
    mask_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "position": self.grid_pos,
            "coordinates": self.world_coords.to_dict(),
            "elevation_range": {"min": self.elevation_range[0], "max": self.elevation_range[1]} if self.elevation_range else None,
            "num_labels": self.num_labels,
            "image": self.image_path,
            "labels": self.labels_path,
        }


def find_elevation_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Find elevation attribute in geodata."""
    for name in ["ELEVATION", "elevation", "ELEV", "elev", "HEIGHT", "height"]:
        if name in gdf.columns:
            return name
    return None


def split_into_tiles(gdf: gpd.GeoDataFrame, rows: int, cols: int) -> list[tuple[int, int, WorldCoordinates, gpd.GeoDataFrame]]:
    """Split geographic data into grid tiles."""
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) / cols
    dy = (maxy - miny) / rows
    
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = minx + c * dx, miny + r * dy
            x2, y2 = x1 + dx, y1 + dy
            
            tile_data = gdf.cx[x1:x2, y1:y2]
            if not tile_data.empty:
                coords = WorldCoordinates(x1, x2, y1, y2)
                tiles.append((r, c, coords, tile_data))
    
    return tiles


def get_random_point(geometry, rng=None):
    """Get random point along a line geometry.
    
    Args:
        geometry: LineString or MultiLineString
        rng: Random number generator (optional, for reproducibility)
    """
    if rng is None:
        rng = random
    
    if isinstance(geometry, LineString):
        line = geometry
    elif isinstance(geometry, MultiLineString):
        line = rng.choice(list(geometry.geoms))
    else:
        return None
    
    distance = rng.uniform(0.2, 0.8) * line.length
    return line.interpolate(distance)


def boxes_overlap(box1, box2, margin=5):
    """Check if two boxes overlap."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 + margin < x2 or x2 + w2 + margin < x1 or
               y1 + h1 + margin < y2 or y2 + h2 + margin < y1)


def render_contour_with_labels(gdf: gpd.GeoDataFrame, output: Path, elev_col: str, size: int, dpi: int) -> list[TextLabel]:
    """Render contour map with non-overlapping elevation labels and natural terrain colors."""
    labels = []
    text_objects = []
    placed_boxes = []
    
    # Get elevation range for color mapping
    elevations = gdf[elev_col].values
    elev_min, elev_max = elevations.min(), elevations.max()
    
    # Create varied terrain colormap with realistic nature colors
    terrain_colors = [
        '#4a90c4',  # Water/very low (blue)
        '#5fa370',  # Low valleys (green)
        '#78b883',  # Grasslands (light green)
        '#9bc78f',  # Plains (pale green)
        '#b8d4a0',  # Rolling hills (yellow-green)
        '#d4c896',  # Foothills (tan)
        '#c8b078',  # Low mountains (brown-tan)
        '#a0896a',  # Mid mountains (brown)
        '#8b7355',  # High mountains (dark brown)
        '#9a9a8f',  # Rocky peaks (gray-brown)
        '#b8b8b0',  # High peaks (light gray)
        '#e0e0d8',  # Snow line (off-white)
        '#f5f5f0'   # Snow peaks (white)
    ]
    terrain_cmap = mcolors.LinearSegmentedColormap.from_list('terrain', terrain_colors)
    
    # Create figure with exact dimensions
    fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
    ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])
    ax.axis('off')
    
    # Create sophisticated terrain background with natural variation
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Create a grid for terrain rendering (balanced resolution for speed)
    grid_res = 200
    x_grid = np.linspace(minx, maxx, grid_res)
    y_grid = np.linspace(miny, maxy, grid_res)
    
    # Create base elevation grid by sampling contours
    Z = np.ones((grid_res, grid_res)) * elev_min
    
    # Use a more efficient approach: rasterize contours with buffering
    unique_elevs = sorted(gdf[elev_col].unique())
    
    for elev in unique_elevs:
        norm_elev = (elev - elev_min) / (elev_max - elev_min) if elev_max > elev_min else 0.5
        contour_geoms = gdf[gdf[elev_col] == elev].geometry
        
        # For each grid point, check if it's near this contour
        for idx, geom in enumerate(contour_geoms):
            try:
                # Buffer the contour
                buffered = geom.buffer(0.002)
                
                # Check grid points within bounds
                geom_minx, geom_miny, geom_maxx, geom_maxy = buffered.bounds
                i_start = int((geom_minx - minx) / (maxx - minx) * grid_res)
                i_end = int((geom_maxx - minx) / (maxx - minx) * grid_res)
                j_start = int((geom_miny - miny) / (maxy - miny) * grid_res)
                j_end = int((geom_maxy - miny) / (maxy - miny) * grid_res)
                
                i_start = max(0, min(grid_res-1, i_start))
                i_end = max(0, min(grid_res, i_end))
                j_start = max(0, min(grid_res-1, j_start))
                j_end = max(0, min(grid_res, j_end))
                
                for i in range(i_start, i_end, 2):  # Sample every 2 for speed
                    for j in range(j_start, j_end, 2):
                        point = Point(x_grid[i], y_grid[j])
                        if buffered.contains(point) or buffered.distance(point) < 0.001:
                            Z[j, i] = elev
            except:
                continue
    
    # Smooth to fill gaps and create natural transitions
    Z = gaussian_filter(Z, sigma=1.5)
    
    # Add multi-scale natural terrain variation
    np.random.seed(hash(str(gdf.total_bounds)) % 10000)  # Unique per tile
    
    # Large-scale variation (geological features)
    noise_large = gaussian_filter(np.random.randn(grid_res, grid_res), sigma=15)
    # Medium-scale variation (hills and valleys)
    noise_medium = gaussian_filter(np.random.randn(grid_res, grid_res), sigma=5)
    # Fine-scale variation (surface texture)
    noise_fine = gaussian_filter(np.random.randn(grid_res, grid_res), sigma=1.5)
    
    # Combine noise at different scales
    combined_noise = (noise_large * 0.15 + noise_medium * 0.10 + noise_fine * 0.05)
    
    # Normalize elevation to [0, 1] with natural variation
    Z_norm = (Z - elev_min) / (elev_max - elev_min) if elev_max > elev_min else np.ones_like(Z) * 0.5
    Z_norm = np.clip(Z_norm + combined_noise, 0, 1)
    
    # Render the terrain with natural color variation
    ax.imshow(Z_norm, extent=[minx, maxx, miny, maxy], origin='lower',
              cmap=terrain_cmap, aspect='auto', interpolation='bicubic', 
              alpha=0.92, zorder=0)
    
    # Draw contour lines on top with elevation-based colors
    for _, row in gdf.iterrows():
        try:
            elev = row[elev_col]
            norm_elev = (elev - elev_min) / (elev_max - elev_min) if elev_max > elev_min else 0.5
            color = terrain_cmap(norm_elev)
            # Darken for better contrast
            line_color = tuple(max(0, c * 0.5) for c in color[:3]) + (1.0,)
            
            gdf[gdf[elev_col] == elev].plot(ax=ax, linewidth=1.3, edgecolor=line_color, facecolor='none', zorder=2)
        except:
            continue
    
    # Add text labels at random points
    for _, row in gdf.iterrows():
        try:
            elev = int(row[elev_col])
            point = get_random_point(row.geometry)
            if point:
                # Normalize elevation for color selection
                norm_elev = (elev - elev_min) / (elev_max - elev_min) if elev_max > elev_min else 0.5
                # Use dark brown text for low elevations, darker for high elevations
                text_color = '#2b1810' if norm_elev < 0.7 else '#1a0f0a'
                
                text_obj = ax.text(
                    point.x, point.y, str(elev),
                    fontsize=9, weight='bold',
                    ha='center', va='center',
                    color=text_color,
                    family='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
                )
                text_objects.append((text_obj, elev))
        except:
            continue
    
    # Render to extract bounding boxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    # Get transform from display pixels to our coordinate system
    transform = fig.transFigure.inverted()
    
    for text_obj, elev in text_objects:
        try:
            # Get bbox in display (pixel) coordinates
            bbox_display = text_obj.get_window_extent(renderer)
            
            # Transform to figure coordinates (0-1)
            bbox_fig = bbox_display.transformed(transform)
            
            # Convert to image pixel coordinates
            x_pix = bbox_fig.x0 * size
            y_pix = (1 - bbox_fig.y1) * size  # Flip Y: figure coords are bottom-up
            w_pix = bbox_fig.width * size
            h_pix = bbox_fig.height * size
            
            box = [x_pix, y_pix, w_pix, h_pix]
            
            # Collision detection
            if not any(boxes_overlap(box, pb) for pb in placed_boxes):
                labels.append(TextLabel(
                    elevation=elev,
                    bbox=box,
                    bbox_normalized=[x_pix/size, y_pix/size, w_pix/size, h_pix/size]
                ))
                placed_boxes.append(box)
            else:
                text_obj.set_visible(False)
        except:
            continue
    
    # Save with exact dimensions
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=dpi, facecolor='white', pad_inches=0)
    plt.close()
    
    logger.info(f"    {len(labels)} labels (filtered from {len(text_objects)})")
    return labels


def render_mask(gdf: gpd.GeoDataFrame, output: Path, size: int, dpi: int):
    """Render binary segmentation mask."""
    fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
    ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])
    ax.axis('off')
    ax.set_facecolor("black")
    gdf.plot(ax=ax, color="white", linewidth=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=dpi, facecolor="black", pad_inches=0)
    plt.close()


class ContourDatasetGenerator:
    """Generate OCR training datasets from DEM contour files."""
    
    def __init__(self, output_dir: Path, tile_size: int = 512, grid_size: int = 20, dpi: int = 150):
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.grid_size = grid_size
        self.dpi = dpi
    
    def process_file(self, input_path: Path, generate_masks: bool = True, generate_annotations: bool = True) -> dict:
        """Process one DEM file into dataset tiles."""
        logger.info(f"Loading {input_path.name}...")
        gdf = gpd.read_file(input_path)
        
        elev_col = find_elevation_column(gdf)
        if elev_col:
            logger.info(f"  Elevation range: {gdf[elev_col].min():.0f}m - {gdf[elev_col].max():.0f}m")
        
        # Create output structure
        name = input_path.stem
        images_dir = self.output_dir / name / "images"
        masks_dir = self.output_dir / name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        if generate_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into tiles
        logger.info(f"  Generating {self.grid_size}x{self.grid_size} grid...")
        tiles = split_into_tiles(gdf, self.grid_size, self.grid_size)
        
        tile_metadata = []
        
        for r, c, coords, tile_gdf in tiles:
            tile_id = f"{name}_r{r:02d}_c{c:02d}"
            img_path = images_dir / f"{tile_id}.png"
            labels_path = images_dir / f"{tile_id}_labels.json"
            
            # Generate labeled image
            labels = render_contour_with_labels(tile_gdf, img_path, elev_col, self.tile_size, self.dpi) if elev_col else []
            
            # Save labels
            labels_data = {
                "image": f"{tile_id}.png",
                "image_size": [self.tile_size, self.tile_size],
                "labels": [l.to_dict() for l in labels]
            }
            with open(labels_path, 'w') as f:
                json.dump(labels_data, f, indent=2)
            
            # Generate mask if requested
            mask_path = None
            if generate_masks:
                mask_path = masks_dir / f"{tile_id}_mask.png"
                render_mask(tile_gdf, mask_path, self.tile_size, self.dpi)
            
            # Metadata
            elev_range = (float(tile_gdf[elev_col].min()), float(tile_gdf[elev_col].max())) if elev_col else None
            
            tile_metadata.append(TileMetadata(
                tile_id=tile_id,
                grid_pos=(r, c),
                world_coords=coords,
                image_path=str(img_path.relative_to(self.output_dir)),
                labels_path=str(labels_path.relative_to(self.output_dir)),
                elevation_range=elev_range,
                num_labels=len(labels),
                mask_path=str(mask_path.relative_to(self.output_dir)) if mask_path else None,
            ))
        
        logger.info(f"  ✓ Generated {len(tile_metadata)} tiles")
        
        # Save metadata
        total_labels = sum(t.num_labels for t in tile_metadata)
        metadata = {
            "source": input_path.name,
            "generated": datetime.now(timezone.utc).isoformat(),
            "config": {"tile_size": self.tile_size, "grid_size": self.grid_size, "dpi": self.dpi},
            "summary": {"total_tiles": len(tile_metadata), "total_labels": total_labels},
            "tiles": [t.to_dict() for t in tile_metadata],
        }
        
        metadata_file = self.output_dir / name / "dataset_summary.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  ✓ {len(tile_metadata)} tiles, {total_labels} labels")
        return metadata
    
    def process_directory(self, input_dir: Path, **kwargs) -> None:
        """Process all DEM files in directory."""
        files = list(input_dir.glob("**/*.shp")) + list(input_dir.glob("**/*.geojson"))
        
        if not files:
            logger.warning(f"No DEM files found in {input_dir}")
            return
        
        logger.info(f"Found {len(files)} file(s)\n")
        
        for file in files:
            self.process_file(file, **kwargs)
        
        logger.info(f"\n✓ Dataset saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate contour map datasets from DEM data")
    parser.add_argument("--input", "-i", required=True, help="Input directory with DEM files")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=512, choices=[50, 256, 512, 1024], help="Tile size in pixels")
    parser.add_argument("--grid-size", type=int, default=3, help="Grid dimensions (NxN)")
    parser.add_argument("--dpi", type=int, default=150, help="Image resolution")
    parser.add_argument("--no-masks", action="store_true", help="Skip mask generation")
    parser.add_argument("--no-annotations", action="store_true", help="Skip annotation extraction")
    
    args = parser.parse_args()
    
    generator = ContourDatasetGenerator(
        output_dir=Path(args.output),
        tile_size=args.tile_size,
        grid_size=args.grid_size,
        dpi=args.dpi,
    )
    
    generator.process_directory(
        Path(args.input),
        generate_masks=not args.no_masks,
        generate_annotations=not args.no_annotations,
    )


if __name__ == "__main__":
    main()
