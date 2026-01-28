# Data Source

1. Download data from [OpenDEM SRTM Download](https://www.opendem.info/srtm_download_contours/)
2. Then use `processData.py`

## Setup Details

Place the downloaded `.shp` or `.geojson` file in `data/dataVisualization/dataExample/` directory.

Example command:
```bash
python src/dataVisualization/processData.py --input data/dataVisualization/dataExample/N63E016/N63E016.shp --output data/dataVisualization/output/example_output
```