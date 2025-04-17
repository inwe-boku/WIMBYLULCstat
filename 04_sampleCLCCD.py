#!/usr/bin/env python3
"""
Process Corine Land Cover (CLC) data by sampling CD tile rasters and computing per-hull statistics.

This script:
- Reads hull polygons (convexhulls) from GeoPackages per country and dataset.
- Reads binary CD tile rasters, clips them, recodes values, and converts to point GeoDataFrames.
- Samples CLC rasters (CLC100 and CLC10) at these points.
- Joins sampled points back to hull polygons and computes area-based CLC statistics per hull.
- Outputs enriched GeoPackages with CLC statistics for each polygon layer.

Usage:
    python process_clc.py [--countries austria,france] [--debug]
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import geometry_mask
from shapely.geometry import Point, box

# -----------------------------------------------------------------------------
# Configuration variables (can be overridden via CLI)
# -----------------------------------------------------------------------------
PROJECT_CRS = "EPSG:3035"
INPUT_CD_TILES_DIR = "data/CDTiles"
INPUT_HULLS_DIR = "data/03_outraster"
INPUT_CLC_DIR = "data/CLC"
OUTPUT_DIR = "data/04_clc"
# DATASETS = ["new", "all", "old"]
DATASETS = ["old"]
CLC_FILES = {
    "clc100": {"file": "U2018_CLC2018_V2020_20u1.tif"},
    "clc10": {"file": "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif"},
}
SAMPLE_LAYER = "convexhulls"
POLY_LAYERS = ["bufferpolys", "concavehulls", "convexhulls"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with 'countries' and 'debug' attributes.
    """
    parser = argparse.ArgumentParser(description="Sample CLC data for turbine hulls per country and dataset.")
    parser.add_argument(
        "--countries",
        type=str,
        help="Comma-separated list of country folder names to process (overrides auto-detection).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    return parser.parse_args()


def get_countries(base_dir: str, override: Optional[List[str]] = None) -> List[str]:
    """
    Determine which country folders to process.

    Args:
        base_dir (str): Path containing country subdirectories.
        override (List[str] or None): Explicit list provided via CLI.

    Returns:
        List[str]: Sorted list of country folder names.
    """
    if override:
        return [c.strip() for c in override]
    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError:
        return []
    return sorted([d for d in entries if os.path.isdir(os.path.join(base_dir, d))])


def read_raster_bounds(cdtile_dir: str) -> pd.DataFrame:
    """
    Read binary CD tile rasters and extract tile names, bounds, and CRS.

    Args:
        cdtile_dir (str): Directory containing binary tile TIFFs.

    Returns:
        pd.DataFrame: Columns ['tile_name', 'bounds', 'crs'].
    """
    records = []
    for fname in sorted(os.listdir(cdtile_dir)):
        if fname.endswith("_binary.tif"):
            path = os.path.join(cdtile_dir, fname)
            with rasterio.open(path) as src:
                bounds = box(*src.bounds)
                name = re.match(r"^([0-9A-Z]+)", fname).group(1)
                records.append({
                    "tile_name": name,
                    "bounds": bounds,
                    "crs": src.crs.to_string(),
                })
    return pd.DataFrame(records)


def process_hulls_and_tiles(
    hulls: gpd.GeoDataFrame,
    tile_df: pd.DataFrame,
    project_crs: str,
    debug: bool = False,
) -> gpd.GeoDataFrame:
    """
    Clip hulls to tiles, select best tiles and compute remainders.

    Args:
        hulls (GeoDataFrame): Polygons with 'tile_best'.
        tile_df (DataFrame): Tiles info ('tile_name','bounds','crs').
        project_crs (str): Desired output CRS.
        debug (bool): If True, print debug info.

    Returns:
        GeoDataFrame: Combined clipped hulls in project_crs.
    """
    # Annotate area and id
    df = hulls.copy()
    df["area"] = df.geometry.area.astype(int)
    df["hull_id"] = df.index
    best_list = []

    # First pass: select polygons matching tile_best
    for _, tile in tile_df.iterrows():
        tn, bnds, tcrs = tile.tile_name, tile.bounds, tile.crs
        in_crs = df.to_crs(tcrs)
        clipped = gpd.overlay(in_crs, gpd.GeoDataFrame({"geometry": [bnds]}, crs=tcrs), how="intersection")
        match = clipped[clipped["tile_best"] == tn].copy()
        if not match.empty:
            match["areac"] = match.geometry.area.astype(int)
            match["areap"] = (match["areac"] / match["area"] * 100).round(2)
            match["tile_sample"] = tn
            best_list.append(match.to_crs(project_crs))
    best_gdf = pd.concat(best_list, ignore_index=True) if best_list else gpd.GeoDataFrame(columns=hulls.columns)

    # Remove best areas from df
    if not best_gdf.empty:
        union_best = best_gdf.buffer(0.5).unary_union
        df.geometry = df.geometry.apply(lambda g: g.difference(union_best))
    else:
        if debug:
            print("No best hulls selected.")

    # Second pass: remainders or full overlaps
    rem_list = []
    for _, tile in tile_df.iterrows():
        tn, bnds, tcrs = tile.tile_name, tile.bounds, tile.crs
        in_crs = df.to_crs(tcrs)
        clipped = gpd.overlay(in_crs, gpd.GeoDataFrame({"geometry": [bnds]}, crs=tcrs), how="intersection")
        if clipped.empty:
            continue
        clipped["areac"] = clipped.geometry.area.astype(int)
        clipped["areap"] = (clipped["areac"] / clipped["area"] * 100).round(2)
        clipped["tile_sample"] = tn
        rem_list.append(clipped.to_crs(project_crs))
    rem_gdf = pd.concat(rem_list, ignore_index=True) if rem_list else gpd.GeoDataFrame(columns=hulls.columns)

    # Combine best and remainders
    combined = pd.concat([best_gdf, rem_gdf], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=project_crs)


def mask_raster_with_polygon_and_recode(
    tile_name: str, raster_path: str, polygon_gdf: gpd.GeoDataFrame
) -> Tuple[np.ndarray, any, any]:
    """
    Mask a binary raster by polygons with matching tile_sample and recode values.

    Args:
        tile_name (str): Tile identifier.
        raster_path (str): Path to binary raster TIFF.
        polygon_gdf (GeoDataFrame): Polygons with 'tile_sample' column.

    Returns:
        masked_raster (ndarray): Recoded array (1->1,0->2), zeros elsewhere.
        transform, crs: rasterio transform and CRS.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        polys = polygon_gdf[polygon_gdf["tile_sample"] == tile_name]
        if polys.empty:
            return np.zeros_like(data), src.transform, src.crs
        mask = geometry_mask(
            polys.to_crs(src.crs).geometry, transform=src.transform, invert=True, out_shape=data.shape, all_touched=True
        )
        out = np.zeros_like(data)
        out[mask & (data == 1)] = 1
        out[mask & (data == 0)] = 2
        return out, src.transform, src.crs


def raster_to_point_gdf(masked_raster: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """
    Convert masked raster array to point GeoDataFrame with CD values.

    Args:
        masked_raster: 2D array of recoded values.
        transform: Affine transform of the raster.
        crs: CRS of raster.

    Returns:
        GeoDataFrame with columns ['geometry','CD'].
    """
    rows, cols = np.where((masked_raster == 1) | (masked_raster == 2))
    vals = masked_raster[rows, cols]
    coords = [rasterio.transform.xy(transform, r, c) for r, c in zip(rows, cols)]
    pts = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame({"geometry": pts, "CD": vals}, crs=crs)


def sample_raster_with_points(raster_path: str, points_gdf: gpd.GeoDataFrame, clc_key: str) -> gpd.GeoDataFrame:
    """
    Sample CLC raster at point locations and attach values.

    Args:
        raster_path (str): Path to CLC raster.
        points_gdf (GeoDataFrame): Points with geometry in project_crs.
        clc_key (str): Key name for output column.

    Returns:
        GeoDataFrame with new column clc_key.
    """
    with rasterio.open(raster_path) as src:
        pts = points_gdf.to_crs(src.crs)
        samples = [val[0] for val in src.sample([(p.x, p.y) for p in pts.geometry])]
    pts[clc_key] = samples
    return pts.to_crs(PROJECT_CRS)


def compute_statistics_by_hull_id(points_gdf: gpd.GeoDataFrame, clc_key: str) -> pd.DataFrame:
    """
    Aggregate CD and VAL counts per hull_id.

    Args:
        points_gdf: Points with 'CD', clc_key, and 'hull_id'.
        clc_key: CLC key e.g. 'clc100'.

    Returns:
        DataFrame with hull_id and count columns.
    """
    groups = points_gdf.groupby("hull_id")
    stats = []
    for hid, grp in groups:
        row = {"hull_id": hid}
        for _, r in grp.iterrows():
            cd = r[clc_key]
            key_val = f"{clc_key}_VAL_{cd}"
            row[key_val] = row.get(key_val, 0) + 1
            if r["CD"] == 1:
                key_cd = f"{clc_key}_CD_{cd}"
                row[key_cd] = row.get(key_cd, 0) + 1
        stats.append(row)
    return pd.DataFrame(stats).fillna(0)


def drop_duplicate_columns_from_dict(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Remove overlapping columns when merging multiple CLC results.

    Args:
        dfs: Dict of DataFrames keyed by clc_key.

    Returns:
        Dict with unique columns per DataFrame.
    """
    seen = set()
    out = {}
    for k, df in dfs.items():
        if not out:
            out[k] = df
            seen.update(df.columns)
        else:
            unique = [c for c in df.columns if c not in seen]
            out[k] = df[unique]
            seen.update(df.columns)
    return out


def combine_clc_statistics(all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge CLC statistics DataFrames on hull_id.

    Args:
        all_results: Dict of {clc_key: DataFrame}.

    Returns:
        Combined DataFrame.
    """
    cleaned = drop_duplicate_columns_from_dict(all_results)
    merged = None
    for df in cleaned.values():
        merged = df if merged is None else merged.merge(df, on="hull_id", how="outer")
    return merged.fillna(0)


def clean_and_sort(hulls: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean temporary columns and sort CLC columns logically.

    Args:
        hulls: GeoDataFrame with CLC stats.

    Returns:
        Reordered GeoDataFrame.
    """
    # Drop intermediate fields
    for col in ["area", "area_percentage"]:
        if col in hulls:
            hulls.drop(columns=[col], inplace=True)
    # Sort CLC columns
    clc_cols = sorted([c for c in hulls.columns if c.startswith("clc")])
    others = [c for c in hulls.columns if c not in clc_cols]
    return hulls[others + clc_cols]


def process_country(country: str, args: argparse.Namespace) -> None:
    """
    Process all datasets for a single country.

    Args:
        country: Country folder name.
        args: Parsed CLI arguments.
    """
    c_low = country.lower()
    in_base = os.path.join(INPUT_HULLS_DIR, c_low)
    out_base = os.path.join(OUTPUT_DIR, c_low)
    os.makedirs(out_base, exist_ok=True)

    for ds in DATASETS:
        hull_path = os.path.join(in_base, f"{c_low}_{ds}.gpkg")
        if not os.path.exists(hull_path):
            if args.debug:
                print(f"Missing hull file: {hull_path}")
            continue

        # Load sample hulls
        hulls = gpd.read_file(hull_path, layer=SAMPLE_LAYER)
        if args.debug:
            print(f"Loaded {len(hulls)} {SAMPLE_LAYER} for {country}-{ds}")

        # Read CD tile bounds
        tile_dir = os.path.join(INPUT_CD_TILES_DIR, country.capitalize())
        tiles = read_raster_bounds(tile_dir)
        combined = process_hulls_and_tiles(hulls, tiles, PROJECT_CRS, debug=args.debug)

        # Mask and convert each tile
        all_pts = []
        for _, t in tiles.iterrows():
            raster_fname = next(
                (f for f in os.listdir(tile_dir) if f.startswith(t.tile_name) and f.endswith("_binary.tif")), None
            )
            if not raster_fname:
                continue
            rast_p = os.path.join(tile_dir, raster_fname)
            masked, tfm, crs = mask_raster_with_polygon_and_recode(t.tile_name, rast_p, combined)
            pts = raster_to_point_gdf(masked, tfm, crs)
            if not pts.empty:
                all_pts.append(pts)
        if not all_pts:
            continue
        points = gpd.GeoDataFrame(pd.concat(all_pts, ignore_index=True)).to_crs(PROJECT_CRS)

        # Sample CLC rasters
        clc_results = {}
        for key, info in CLC_FILES.items():
            path = os.path.join(INPUT_CLC_DIR, info["file"])
            if args.debug:
                print(f"Sampling {key} from {path}")
            pts = sample_raster_with_points(path, points, key)
            clc_results[key] = compute_statistics_by_hull_id(pts, key)
        stats = combine_clc_statistics(clc_results)

        # Attach stats to each layer and save
        for layer in POLY_LAYERS:
            gdf = gpd.read_file(hull_path, layer=layer)
            gdf["hull_id"] = gdf.index
            merged = gdf.merge(stats, on="hull_id", how="left").fillna(0)
            merged = merged.to_crs(PROJECT_CRS)
            merged = clean_and_sort(merged)
            out_path = os.path.join(out_base, f"{c_low}_{ds}.gpkg")
            merged.to_file(out_path, layer=layer, driver="GPKG")


def main():
    """
    Entry point: parse CLI args, determine countries, and process each.
    """
    args = parse_args()
    override = args.countries.split(",") if args.countries else None
    countries = get_countries(INPUT_HULLS_DIR, override)
    if not countries:
        print(f"No countries found in '{INPUT_HULLS_DIR}'. Exiting.")
        return
    for country in countries:
        if args.debug:
            print(f"Processing country: {country}")
        process_country(country, args)


if __name__ == "__main__":
    main()
