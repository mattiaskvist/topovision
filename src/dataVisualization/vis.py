"""Previous!! Not Really Nessessary Now...."""


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("dataExample") # Example
OUTPUT_DIR = Path("output")
GRID_SIZE = 50
HEIGHT_COLUMNS = ["elevation", "ELEV", "elev", "HEIGHT", "height", "CONTOUR", "contour"]
DPI = 200
LINEWIDTH = 0.5
CMAP = "viridis"  


def find_height_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Find the elevation/height column in the GeoDataFrame."""
    for col in HEIGHT_COLUMNS:
        if col in gdf.columns:
            return col
    return None


def get_grid_bounds(gdf: gpd.GeoDataFrame, rows: int, cols: int) -> list[tuple]:
    """Calculate grid cell bounds for splitting the data."""
    minx, miny, maxx, maxy = gdf.total_bounds
    x_step = (maxx - minx) / cols
    y_step = (maxy - miny) / rows
    
    bounds = []
    for row in range(rows):
        for col in range(cols):
            cell_bounds = (
                minx + col * x_step,
                miny + row * y_step,
                minx + (col + 1) * x_step,
                miny + (row + 1) * y_step,
            )
            bounds.append((row, col, cell_bounds))
    return bounds


def plot_contours(
    gdf: gpd.GeoDataFrame,
    height_col: str | None,
    title: str,
    output_path: Path,
    vmin: float = None,
    vmax: float = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if height_col:
        gdf.plot(
            column=height_col,
            ax=ax,
            linewidth=LINEWIDTH,
            legend=True,
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        gdf.plot(ax=ax, linewidth=LINEWIDTH)
    
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def process_shapefile(shp_path: Path) -> None:
    """Process a single shapefile: load, split into grid, and save visualizations."""
    print(f"Loading {shp_path.name}...")
    gdf = gpd.read_file(shp_path)
    
    height_col = find_height_column(gdf)
    if height_col:
        print(f"  Found height column: {height_col}")
        vmin, vmax = gdf[height_col].min(), gdf[height_col].max()
        print(f"  Elevation range: {vmin:.1f} - {vmax:.1f}")
    else:
        print("  No height column found")
        vmin, vmax = None, None
    
    # Create output directory for this shapefile
    tile_name = shp_path.stem
    tile_output_dir = OUTPUT_DIR / tile_name
    tile_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate grid cells and plot each
    grid_bounds = get_grid_bounds(gdf, GRID_SIZE, GRID_SIZE)
    
    for row, col, (minx, miny, maxx, maxy) in grid_bounds:
        # Clip data to grid cell
        cell_gdf = gdf.cx[minx:maxx, miny:maxy]
        
        if cell_gdf.empty:
            continue
        
        output_path = tile_output_dir / f"{tile_name}_r{row}_c{col}.png"
        title = f"{tile_name} (Row {row}, Col {col})"
        
        plot_contours(cell_gdf, height_col, title, output_path, vmin, vmax)
        print(f"  Saved {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    shapefiles = sorted(DATA_DIR.glob("**/*.shp"))
    print(f"Found {len(shapefiles)} shapefiles\n")
    
    for shp in shapefiles:
        process_shapefile(shp)
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()
