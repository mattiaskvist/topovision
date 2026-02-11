#!/usr/bin/env python3
"""Simple runner: load a HeightExtractionOutput JSON and export an OBJ."""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(repo_root))

from src.height_extraction.schemas import HeightExtractionOutput
from src.height_extraction.mesh_generation import generate_heightmap, export_to_obj


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print("Usage: run_mesh_from_json.py <input.json> <output.obj>")
        return 2

    input_path = Path(argv[0])
    output_path = Path(argv[1])

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 2

    data = json.loads(input_path.read_text(encoding="utf-8"))
    output = HeightExtractionOutput.parse_obj(data)

    grid_x, grid_y, grid_z = generate_heightmap(output, resolution_scale=1.0, interpolation_method="cubic", smoothing_sigma=0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_obj(grid_x, grid_y, grid_z, str(output_path), scale_z=1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
