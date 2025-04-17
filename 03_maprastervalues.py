#!/usr/bin/env python3
"""
Process hull and turbine layers to sample raster attributes per country and dataset.

This script:
- Reads GeoPackage layers (points and polygons) for each country and dataset.
- Samples raster values for points (direct cell lookup) and polygons (mean over pixels).
- Supports multiple raster fields (e.g., altitude, slope, TRI) with configurable precision.
- Writes updated layers back to output GeoPackages.

Usage:
    python process_rasters.py [--countries austria,poland] [--debug]
"""

import argparse
import os

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration (override via CLI)
# -----------------------------------------------------------------------------
INPUT_DIR = "data/02_outhulls"
OUTPUT_DIR = "data/03_outraster"
RASTER_DIR = "data/RASTER"
DATASETS = ["new", "all", "old"]
RASTER_FILES = {
    "altitude": {"file": "Copernicus_DSM_90m_COG_3035.tif", "precision": 0},
    "slope": {"file": "Copernicus_SLOPE_90m_COG_3035.tif", "precision": 1},
    "tri": {"file": "Copernicus_TRI_90m_COG_3035.tif", "precision": 2},
}
POLYTYPES = ["turbines", "bufferpolys", "concavehulls", "convexhulls"]


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: with attributes 'countries' (list[str]) and 'debug' (bool).
    """
    parser = argparse.ArgumentParser(description="Sample raster values for turbine/hull layers per country.")
    parser.add_argument(
        "--countries",
        type=str,
        help="Comma-separated list of country codes to process (overrides auto-detection).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for missing files or sampling errors.",
    )
    return parser.parse_args()


def get_countries(input_dir, override_list=None):
    """
    Determine which country folders to process.

    Args:
        input_dir (str): Path containing country subdirectories.
        override_list (list[str] or None): Explicit list of country names.

    Returns:
        list[str]: Sorted list of country folder names.
    """
    if override_list:
        return [c.strip() for c in override_list]
    try:
        entries = os.listdir(input_dir)
    except FileNotFoundError:
        return []
    return sorted([d for d in entries if os.path.isdir(os.path.join(input_dir, d))])


def calculate_mean_raster_value(raster, geom, pixel_size=None, debug=False):
    """
    Sample the mean raster value under a polygon geometry.

    If all pixels are nodata, optionally buffer by half a pixel and retry.

    Args:
        raster (rasterio.DatasetReader): Opened rasterio dataset.
        geom (shapely.geometry.BaseGeometry): Polygon geometry to sample.
        pixel_size (float or None): Raster pixel size, if buffering needed.
        debug (bool): If True, print masking errors.

    Returns:
        float: Mean raster value or np.nan.
    """
    try:
        masked, _ = rasterio.mask.mask(raster, [geom], crop=True, filled=True, nodata=np.nan)
        data = masked[0]
        if np.isnan(data).all() and pixel_size:
            buffered = geom.buffer(pixel_size / 2)
            masked, _ = rasterio.mask.mask(raster, [buffered], crop=True, filled=True, nodata=np.nan)
            data = masked[0]
        if np.isnan(data).all():
            return np.nan
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(data))
    except Exception as e:
        if debug:
            print(f"Error sampling raster: {e}")
        return np.nan


def process_country(country, args):
    """
    Process all datasets and rasters for a single country.

    Reads GPKG layers, samples rasters, and writes outputs.

    Args:
        country (str): Country folder name.
        args (argparse.Namespace): Parsed CLI arguments.
    """
    in_dir = os.path.join(INPUT_DIR, country)
    out_dir = os.path.join(OUTPUT_DIR, country)
    os.makedirs(out_dir, exist_ok=True)

    total_tasks = len(DATASETS) * len(RASTER_FILES) * len(POLYTYPES)
    with tqdm(total=total_tasks, desc=f"Processing {country}", leave=False) as progress:
        for ds in DATASETS:
            gpkg_path = os.path.join(in_dir, f"{country}_{ds}.gpkg")
            if not os.path.exists(gpkg_path):
                if args.debug:
                    print(f"[WARNING] Missing file: {gpkg_path}")
                continue

            # Read layers and compute combined bounding box
            gdfs = {}
            bbox = None
            for layer in POLYTYPES:
                try:
                    gdf = gpd.read_file(gpkg_path, layer=layer)
                    gdfs[layer] = gdf
                    bounds = gdf.total_bounds
                    if bbox is None:
                        bbox = bounds.copy()
                    else:
                        bbox = [
                            min(bbox[0], bounds[0]),
                            min(bbox[1], bounds[1]),
                            max(bbox[2], bounds[2]),
                            max(bbox[3], bounds[3]),
                        ]
                except Exception as e:
                    if args.debug:
                        print(f"Error reading {layer} in {gpkg_path}: {e}")
                progress.update(1)

            if bbox is None:
                if args.debug:
                    print(f"[INFO] No valid layers in {gpkg_path}.")
                continue

            # Process each raster field
            for field, info in RASTER_FILES.items():
                rast_path = os.path.join(RASTER_DIR, info["file"])
                precision = info["precision"]
                with rasterio.open(rast_path) as rast:
                    pixsize = rast.res[0]
                    window = rast.window(*bbox)
                    raster_data = rast.read(1, window=window, masked=True)

                    # Sample each layer
                    for layer, gdf in gdfs.items():
                        # Reproject if needed
                        if gdf.crs != rast.crs:
                            gdf = gdf.to_crs(rast.crs)

                        geom_types = gdf.geometry.geom_type.unique()
                        if len(geom_types) > 1:
                            raise ValueError(f"Mixed geometry types in {layer}: {geom_types}")
                        geom_type = geom_types[0].lower()

                        if geom_type == "point":
                            values = []
                            for pt in gdf.geometry:
                                try:
                                    r, c = rast.index(pt.x, pt.y)
                                    r_off = int(r - window.row_off)
                                    c_off = int(c - window.col_off)
                                    if 0 <= r_off < raster_data.shape[0] and 0 <= c_off < raster_data.shape[1]:
                                        val = float(raster_data[r_off, c_off])
                                        if precision == 0:
                                            values.append(int(val))
                                        else:
                                            values.append(round(val, precision))
                                    else:
                                        values.append(np.nan)
                                except Exception:
                                    values.append(np.nan)
                            gdf[field] = values

                        elif geom_type in ("polygon", "multipolygon"):

                            def _sample_poly(geom):
                                val = calculate_mean_raster_value(rast, geom, pixel_size=pixsize, debug=args.debug)
                                if np.isnan(val):
                                    return np.nan
                                return int(val) if precision == 0 else round(val, precision)

                            gdf[field] = gdf.geometry.apply(_sample_poly)

                        else:
                            if args.debug:
                                print(f"Unsupported geometry {geom_type} in {layer}")

                        gdfs[layer] = gdf
                        progress.update(1)

            # Write back updated layers
            for layer, gdf in gdfs.items():
                out_path = os.path.join(out_dir, f"{country}_{ds}.gpkg")
                gdf.to_file(out_path, layer=layer, driver="GPKG")
                if args.debug:
                    print(f"Wrote {len(gdf)} features to {layer} in {out_path}")
                progress.update(1)


def main():
    """
    Entry point: parse arguments, determine countries, and process each country.
    """
    args = parse_args()
    override = args.countries.split(",") if args.countries else None
    countries = get_countries(INPUT_DIR, override)
    if not countries:
        print(f"No country folders found in '{INPUT_DIR}'. Exiting.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for country in tqdm(countries, desc="Countries"):
        process_country(country, args)


if __name__ == "__main__":
    main()
