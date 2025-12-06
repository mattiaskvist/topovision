"""Dataset generation examples."""

import sys
from pathlib import Path

import geopandas as gpd

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataHelpers import ContourDatasetGenerator, find_elevation_column


def example_quick_test():
    """Quick test - 20x20 grid."""
    print("=== Quick Test (20x20 grid) ===\n")
    
    generator = ContourDatasetGenerator(
        output_dir=SCRIPT_DIR / "output" / "quick_test",
        tile_size=512,
        grid_size=20,  
        dpi=150,
    )
    
    data_path = SCRIPT_DIR / "dataExample" / "N63E016" / "N63E016.shp"
    generator.process_file(data_path)


def example_50x50_tiles():
    """Generate 50x50 pixel tiles."""
    print("\n=== 50x50 Pixel Tiles ===\n")
    
    generator = ContourDatasetGenerator(
        output_dir=SCRIPT_DIR / "output" / "tiles_50x50",
        tile_size=50,
        grid_size=20,  
        dpi=100,
    )
    
    data_path = SCRIPT_DIR / "dataExample" / "N63E016" / "N63E016.shp"
    generator.process_file(data_path)


def example_high_resolution():
    """High-res dataset."""
    print("\n=== High Resolution ===\n")
    
    generator = ContourDatasetGenerator(
        output_dir=SCRIPT_DIR / "output" / "high_res",
        tile_size=1024,
        grid_size=8, 
        dpi=200,
    )
    
    data_path = SCRIPT_DIR / "dataExample" / "N63E016" / "N63E016.shp"
    generator.process_file(data_path, generate_masks=True, generate_annotations=True)


def example_masks_only():
    """Masks only."""
    print("\n=== Masks Only ===\n")
    
    generator = ContourDatasetGenerator(
        output_dir=SCRIPT_DIR / "output" / "masks_only",
        tile_size=512,
        grid_size=3,
    )
    
    data_path = SCRIPT_DIR / "dataExample" / "N63E016" / "N63E016.shp"
    generator.process_file(data_path, generate_masks=True, generate_annotations=False)


def example_inspect_data():
    """Inspect source data."""
    print("\n=== Data Inspection ===\n")
    
    data_path = SCRIPT_DIR / "dataExample" / "N63E016" / "N63E016.shp"
    gdf = gpd.read_file(data_path)
    
    print(f"File: {data_path.name}")
    print(f"Features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    
    elev_col = find_elevation_column(gdf)
    if elev_col:
        print(f"Elevation: {gdf[elev_col].min():.0f}m - {gdf[elev_col].max():.0f}m")
    
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"Bounds: [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]")
    print(f"Columns: {list(gdf.columns)}\n")


def example_batch_processing():
    """Batch process directory."""
    print("\n=== Batch Processing ===\n")
    
    generator = ContourDatasetGenerator(
        output_dir=SCRIPT_DIR / "output" / "batch",
        tile_size=512,
        grid_size=3,
    )
    
    input_dir = SCRIPT_DIR / "dataExample"
    generator.process_directory(input_dir, generate_masks=True, generate_annotations=True)


if __name__ == "__main__":
    print("Contour Dataset Generator Examples")
    print("=" * 50 + "\n")
    
    # Run quick test
    example_inspect_data()
    example_quick_test()
    
    # Uncomment to run other examples:
    # example_50x50_tiles()
    # example_high_resolution()
    # example_masks_only()
    # example_batch_processing()
