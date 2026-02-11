# Data Source

1. Download data from [OpenDEM SRTM Download](https://www.opendem.info/srtm_download_contours/)
2. Then use `process_data.py` to generate training tiles.

## Setup Details

Place the downloaded `.shp` or `.geojson` file in `data/` (for example,
`data/N63E016/N63E016.shp`).

Example command:

```bash
uv run python src/data_pipeline/process_data.py \
  --input data/N63E016/N63E016.shp \
  --output data/training
```
