#!/usr/bin/env python3
"""
Script for processing Corine Land Cover (CLC) data for specified countries,
with progress indicators for tile‐masking and per‐layer hull processing.

Usage:
    python process_clc.py --countries austria,france
"""

import argparse
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point, box
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEBUG = True
PROJECT_CRS = "EPSG:3035"
CLC_FILES = {
    "clc100": {"file": "U2018_CLC2018_V2020_20u1.tif"},
    "clc10": {"file": "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif"},
}
DATASETS = ["new", "old", "all"]
SAMPLE_LAYER = "convexhulls"
LAYERS = ["bufferpolys", "concavehulls", "convexhulls"]


def read_raster_bounds(cdtile_path):
    """
    Read binary raster tile files from a directory and extract their bounds,
    CRS, and a 'channel' identifier from the filename.

    Returns a DataFrame with columns:
      - tile_name: e.g. '29UNA'
      - channel:   e.g. 'ch3lt0'
      - bounds:    shapely Polygon
      - crs:       CRS string
    """
    tile_data = []
    for fn in sorted(os.listdir(cdtile_path)):
        if fn.endswith("_binary.tif"):
            path = os.path.join(cdtile_path, fn)
            with rasterio.open(path) as src:
                bounds = box(*src.bounds)
                tile_id = re.match(r"^([0-9A-Z]+)", fn).group(1)
                pattern = re.compile(r"^[0-9A-Z]+_([A-Za-z0-9_]+)_binary\.tif$")
                channel = pattern.match(fn).group(1)
                tile_crs = src.crs.to_string()
                tile_data.append({"tile_name": tile_id, "channel": channel, "bounds": bounds, "crs": tile_crs})
    return pd.DataFrame(tile_data)


def process_hulls_and_tiles(hulls: gpd.GeoDataFrame, tile_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Clip hull polygons to each tile, compute clipped areas, subtract the best clips,
    then clip the remainder. Returns all clipped parts in PROJECT_CRS.
    """
    hulls = hulls.copy()
    hulls["area"] = hulls.geometry.area.astype(int)
    hulls["hull_id"] = hulls.index
    best_clips = []

    # First pass: gather the 'best' portions
    for _, tile in tile_df.iterrows():
        name, bounds, crs = tile["tile_name"], tile["bounds"], tile["crs"]
        in_crs = hulls.to_crs(crs)
        poly = gpd.GeoDataFrame({"geometry": [bounds]}, crs=crs)
        clipped = gpd.overlay(in_crs, poly, how="intersection")
        keep = []
        for _, part in clipped.iterrows():
            orig = hulls.loc[part["hull_id"]]
            if orig["tile_best"] == name:
                a = part.geometry.area
                pct = (a / orig["area"] * 100) if orig["area"] > 0 else 0
                part["areac"] = int(a)
                part["areap"] = round(pct, 2)
                part["tile_sample"] = name
                keep.append(part)
        if keep:
            df = pd.DataFrame(keep)
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
            best_clips.append(gdf.to_crs(PROJECT_CRS))

    # Subtract best clips from original hulls
    if best_clips:
        best = gpd.GeoDataFrame(pd.concat(best_clips, ignore_index=True), crs=PROJECT_CRS)
        best.to_file("data/debug/hulls_best.gpkg")
        diff = best.to_crs(hulls.crs).buffer(0.5).union_all()
        hulls.geometry = hulls.geometry.apply(lambda g: g.difference(diff))
    else:
        best = gpd.GeoDataFrame(columns=hulls.columns, crs=PROJECT_CRS)

    hulls = hulls[~hulls.geometry.is_empty]
    hulls.to_file("data/debug/hulls_remains.gpkg")

    # Second pass: clip what's left
    rem_clips = []
    best_ids = set(best["hull_id"])
    for _, tile in tile_df.iterrows():
        name, bounds, crs = tile["tile_name"], tile["bounds"], tile["crs"]
        in_crs = hulls.to_crs(crs)
        poly = gpd.GeoDataFrame({"geometry": [bounds]}, crs=crs)
        clipped = gpd.overlay(in_crs, poly, how="intersection")
        keep = []
        for _, part in clipped.iterrows():
            orig = hulls.loc[part["hull_id"]]
            a = part.geometry.area
            pct = (a / orig["area"] * 100) if orig["area"] > 0 else 0
            part["areac"] = int(a)
            part["areap"] = round(pct, 2)
            if part["hull_id"] in best_ids or pct > 99.9:
                part["tile_sample"] = name
                keep.append(part)
        if keep:
            df = pd.DataFrame(keep)
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
            rem_clips.append(gdf.to_crs(PROJECT_CRS))

    if rem_clips:
        rem = gpd.GeoDataFrame(pd.concat(rem_clips, ignore_index=True), crs=PROJECT_CRS)
        rem.to_file("data/debug/hulls_remain_clipped.gpkg")
        return gpd.GeoDataFrame(pd.concat([best, rem], ignore_index=True))
    return best


def mask_raster_with_polygon_and_recode(tile_name, raster_path, polygon_gdf):
    """
    Mask a binary raster with polygons (for this tile_name), then recode:
     1→1, 0→2. Returns (array, transform, crs).
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        trans = src.transform
        crs = src.crs
        polys = polygon_gdf[polygon_gdf["tile_sample"] == tile_name]
        if polys.empty:
            return np.zeros_like(arr), trans, crs
        polys = polys.to_crs(crs)
        mask = geometry_mask(polys.geometry, transform=trans, invert=True, out_shape=arr.shape, all_touched=True)
        out = np.zeros_like(arr)
        out[mask & (arr == 1)] = 1
        out[mask & (arr == 0)] = 2
    return out, trans, crs


def raster_to_point_gdf(masked, transform, crs):
    """
    Convert masked array to points with CD field (1 or 2).
    """
    rows, cols = np.where((masked == 1) | (masked == 2))
    vals = masked[rows, cols]
    coords = [rasterio.transform.xy(transform, r, c) for r, c in zip(rows, cols)]
    pts = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame({"geometry": pts, "CD": vals}, crs=crs)


def sample_raster_with_points(raster_path, points_gdf, clc_key):
    """
    Sample LULC raster at each point, write result into clc_key column.
    """
    with rasterio.open(raster_path) as src:
        pts = points_gdf.to_crs(src.crs)
        vals = [v[0] for pt in pts.geometry for v in src.sample([(pt.x, pt.y)])]
    points_gdf[clc_key] = vals
    return points_gdf.to_crs(PROJECT_CRS)


def compute_statistics_by_hull_id(points_gdf, clc_key):
    """
    For each hull_id group:
      - count {clc_key}_CD_val where CD==1
      - count {clc_key}_VAL_val for all points
    Shows a tqdm bar for hulls.
    """
    for fld in ["CD", clc_key, "hull_id"]:
        if fld not in points_gdf:
            raise ValueError(f"{fld} missing")

    grouped = points_gdf.groupby("hull_id")
    stats_list = []
    for hull_id, group in tqdm(grouped, desc=f"Stats {clc_key}", total=grouped.ngroups, unit="hull"):
        stats = {}
        for _, row in group.iterrows():
            if row["CD"] == 1:
                key_cd = f"{clc_key}_CD_{row[clc_key]}"
                stats[key_cd] = stats.get(key_cd, 0) + 1
            key_val = f"{clc_key}_VAL_{row[clc_key]}"
            stats[key_val] = stats.get(key_val, 0) + 1
        stats["hull_id"] = hull_id
        stats_list.append(stats)

    return pd.DataFrame(stats_list).fillna(0)


def drop_duplicate_columns_from_dict(dfs):
    seen, out = set(), {}
    for k, df in dfs.items():
        if not out:
            out[k] = df
            seen.update(df.columns)
        else:
            uniq = [c for c in df.columns if c not in seen]
            out[k] = df[uniq]
            seen.update(df.columns)
    return out


def combine_clc_statistics(all_results):
    merged = None
    clean = drop_duplicate_columns_from_dict(all_results)
    for df in clean.values():
        merged = df if merged is None else merged.merge(df, left_index=True, right_index=True, how="outer")
    return merged


def parse_arguments():
    p = argparse.ArgumentParser(description="Process CLC data for specified countries.")
    p.add_argument("--countries", type=str, required=True, help="Comma-separated list, e.g. austria,france")
    return p.parse_args()


def clean_and_sort(hulls):
    if "cluster" not in hulls and "cls_new" in hulls:
        hulls.rename(columns={"cls_new": "cluster"}, inplace=True)
    if "buffer" not in hulls and "buf_new" in hulls:
        hulls.rename(columns={"buf_new": "buffer"}, inplace=True)
    drops = [c for c in ["split_area", "original_area", "area_percentage"] if c in hulls]
    hulls.drop(columns=drops, inplace=True)
    clc_cols = sorted(
        [c for c in hulls if c.startswith("clc")],
        key=lambda x: (x.split("_")[0], int(x.split("_")[-1]), x.split("_")[1]),
    )
    others = [c for c in hulls if not c.startswith("clc")]
    return hulls[others + clc_cols]


def main():
    args = parse_arguments()
    countries = args.countries.split(",")

    for country in countries:
        for ds in DATASETS:
            cap = country.capitalize()
            tile_dir = f"data/CDTiles/{cap}/"
            in_gpkg = f"data/03_outraster/{country}/{country}_{ds}.gpkg"
            out_gpkg = f"data/04_clc/{country}/{country}_{ds}.gpkg"
            os.makedirs(os.path.dirname(out_gpkg), exist_ok=True)

            if DEBUG:
                print(f"Processing {country}/{ds}")

            # Read tile metadata and build a GeoDataFrame of tile polygons in PROJECT_CRS
            tiles = read_raster_bounds(tile_dir)
            tpolys = []
            for _, t in tiles.iterrows():
                gdf = gpd.GeoDataFrame({"channel": [t["channel"]]}, geometry=[t["bounds"]], crs=t["crs"])
                tpolys.append(gdf.to_crs(PROJECT_CRS))
            tile_polys = gpd.GeoDataFrame(pd.concat(tpolys, ignore_index=True), crs=PROJECT_CRS)

            # Process hulls
            hulls = gpd.read_file(in_gpkg, layer=SAMPLE_LAYER)
            combined = process_hulls_and_tiles(hulls, tiles)

            # Mask each tile, convert to points, tag with channel
            pts_list = []
            total_tiles = len(tiles)
            for idx, t in enumerate(tiles.itertuples(index=False), start=1):
                tname, chan = t.tile_name, t.channel
                match = next(
                    (f for f in os.listdir(tile_dir) if f.startswith(tname) and f.endswith("binary.tif")), None
                )
                if not match:
                    if DEBUG:
                        print(f"Skipping missing raster for tile {tname} ({idx}/{total_tiles})")
                    continue

                if DEBUG:
                    print(f"Masking tile {idx}/{total_tiles}: {tname}, channel={chan}")

                mr, tf, cr = mask_raster_with_polygon_and_recode(tname, os.path.join(tile_dir, match), combined)
                pts = raster_to_point_gdf(mr, tf, cr)
                if pts.empty:
                    continue
                pts = pts.to_crs(PROJECT_CRS)
                pts["channel"] = chan
                pts_list.append(pts)

            all_pts = gpd.GeoDataFrame(pd.concat(pts_list, ignore_index=True), crs=PROJECT_CRS)

            # Sample CLC rasters
            for key, info in CLC_FILES.items():
                path = f"data/CLC/{info['file']}"
                if DEBUG:
                    print(f"Sampling {key}")
                all_pts = sample_raster_with_points(path, all_pts, key)

            # Write turbines layer with channel join
            tur = gpd.read_file(in_gpkg, layer="turbines").to_crs(PROJECT_CRS)
            tur = gpd.sjoin(tur, tile_polys[["channel", "geometry"]], how="left", predicate="within").drop(
                columns=["index_right"]
            )
            tur.to_file(out_gpkg, layer="turbines", driver="GPKG")

            # Process each polygon layer with tqdm
            for lyr in tqdm(LAYERS, desc="Polygon layers", unit="layer"):
                if DEBUG:
                    print(f"Processing layer: {lyr}")
                hs = gpd.read_file(in_gpkg, layer=lyr).to_crs(PROJECT_CRS)
                hs["hull_id"] = hs.index

                joined = gpd.sjoin(all_pts, hs[["hull_id", "geometry"]], how="left", predicate="within")

                # Compute stats per hull with per‐hull tqdm bars
                all_stats = pd.DataFrame()
                for key in CLC_FILES:
                    df = compute_statistics_by_hull_id(joined, key)
                    all_stats = df if all_stats.empty else all_stats.merge(df, on="hull_id")

                # Determine modal channel per hull
                channel_df = (
                    joined.groupby("hull_id")["channel"]
                    .agg(lambda x: x.mode()[0] if not x.mode().empty else "")
                    .reset_index()
                )

                out = hs.merge(all_stats, on="hull_id", how="left").fillna(0)
                out = out.merge(channel_df, on="hull_id", how="left")
                out["channel"] = out["channel"].fillna("")
                out["area"] = out.geometry.area.astype(int)
                out = clean_and_sort(out)
                out.to_file(out_gpkg, layer=lyr, driver="GPKG")


if __name__ == "__main__":
    main()
