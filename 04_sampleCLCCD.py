import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

# Silence FutureWarning about downcasting on fillna(): opt into future behavior
pd.set_option("future.no_silent_downcasting", True)
import rasterio
import tyro
from joblib import Parallel, delayed
from rasterio import mask
from rasterio.features import geometry_mask
from shapely.geometry import Point, box
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEBUG = True
PROJECT_CRS = "EPSG:3035"
# CLC files + optional upscale factors (integer): number of 10m pixels per original pixel
CLC_FILES = {
    "clc100": {"file": "U2018_CLC2018_V2020_20u1.tif", "upscale": 10},
    "clc10": {"file": "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif", "upscale": 1},
}
SAMPLE_LAYER = "convexhulls"
LAYERS = ["bufferpolys", "concavehulls", "convexhulls"]

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


def read_raster_bounds(cdtile_path):
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
                tile_data.append({
                    "tile_name": tile_id,
                    "channel": channel,
                    "bounds": bounds,
                    "crs": tile_crs,
                })
    return pd.DataFrame(tile_data)


def process_hulls_and_tiles(hulls: gpd.GeoDataFrame, tile_df: pd.DataFrame) -> gpd.GeoDataFrame:
    hulls = hulls.copy()
    hulls["area"] = hulls.geometry.area.astype(int)
    hulls["hull_id"] = hulls.index
    if "tile_best" not in hulls.columns:
        hulls["tile_best"] = ""
    best_clips = []

    # First pass: best portions
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

    if best_clips:
        best = gpd.GeoDataFrame(pd.concat(best_clips, ignore_index=True), crs=PROJECT_CRS)
        best.to_file("data/debug/hulls_best.gpkg")
        diff = best.to_crs(hulls.crs).buffer(0.5).union_all()
        hulls.geometry = hulls.geometry.apply(lambda g: g.difference(diff))
    else:
        best = gpd.GeoDataFrame(columns=hulls.columns, crs=PROJECT_CRS)

    hulls = hulls[~hulls.geometry.is_empty]
    hulls.to_file("data/debug/hulls_remains.gpkg")

    # Second pass: remaining portions
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
    rows, cols = np.where((masked == 1) | (masked == 2))
    vals = masked[rows, cols]
    coords = [rasterio.transform.xy(transform, r, c) for r, c in zip(rows, cols)]
    pts = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame({"geometry": pts, "CD": vals}, crs=crs)


def sample_raster_with_points(raster_path, points_gdf, clc_key):
    with rasterio.open(raster_path) as src:
        pts = points_gdf.to_crs(src.crs)
        vals = [v[0] for pt in pts.geometry for v in src.sample([(pt.x, pt.y)])]
    points_gdf[clc_key] = vals
    return points_gdf.to_crs(PROJECT_CRS)


def compute_statistics_by_hull_id(points_gdf, clc_key):
    for fld in ["CD", clc_key, "hull_id"]:
        if fld not in points_gdf:
            raise ValueError(f"{fld} missing")
    grouped = points_gdf.groupby("hull_id")
    stats_list = []
    for hull_id, group in tqdm(grouped, desc=f"Stats {clc_key}", total=grouped.ngroups, unit="hull"):
        stats = {"hull_id": hull_id}
        for _, row in group.iterrows():
            if row["CD"] == 1:
                key_cd = f"{clc_key}_CD_{row[clc_key]}"
                stats[key_cd] = stats.get(key_cd, 0) + 1
            key_val = f"{clc_key}_VAL_{row[clc_key]}"
            stats[key_val] = stats.get(key_val, 0) + 1
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


def _process_single_hull(idx, geom, clc_key, filepath, upscale):
    """
    Sample a single hull polygon against the CLC raster, applying an optional upscaling
    by integer factor (replicates each pixel into a block).
    """
    # Open the raster and extract the masked window for this polygon
    with rasterio.open(filepath) as src:
        nodata_val = src.nodata
        out_image, _ = mask.mask(src, [geom], crop=True)
        band = out_image[0]

    # Apply integer upscaling if requested
    if upscale > 1:
        band = np.repeat(np.repeat(band, upscale, axis=0), upscale, axis=1)

    # Flatten and filter out nodata and zero values
    vals = band.ravel()
    if nodata_val is not None:
        vals = vals[vals != nodata_val]
    vals = vals[vals != 0]

    # Compute counts of each class
    if vals.size > 0:
        counts = pd.Series(vals).value_counts().to_dict()
    else:
        counts = {}

    # Build stats dict: CD always zero, VAL counts from counts
    stat = {f"{clc_key}_CD_{v}": 0 for v in counts}
    for v, cnt in counts.items():
        stat[f"{clc_key}_VAL_{v}"] = cnt
    stat["hull_id"] = idx
    return stat


def sample_only_clc(hulls: gpd.GeoDataFrame, clc_files: dict, n_jobs: int = -1) -> pd.DataFrame:
    """
    Parallel sampling of CLC rasters per hull polygon with optional upscaling. CD counts set to zero.
    """
    all_stats = []
    for clc_key, info in clc_files.items():
        filepath = f"data/CLC/{info['file']}"
        upscale = info.get("upscale", 1)
        # parallel sampling
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_hull)(idx, hull.geometry, clc_key, filepath, upscale)
            for idx, hull in hulls.iterrows()
        )
        df = pd.DataFrame(results).fillna(0).infer_objects(copy=False)
        all_stats.append(df)
    # merge stats per hull
    merged = all_stats[0]
    for df in all_stats[1:]:
        merged = merged.merge(df, on="hull_id", how="outer")
    return merged.fillna(0).infer_objects(copy=False)


def main(countries: list[str]):
    for country in countries:
        in_dir = f"data/03_outraster/{country}/"
        out_dir = f"data/04_clc/{country}/"
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(in_dir):
            if not fname.endswith(".gpkg"):
                continue
            ds = fname.replace(f"{country}_", "").replace(".gpkg", "")
            in_gpkg = os.path.join(in_dir, fname)
            out_gpkg = os.path.join(out_dir, fname)

            if DEBUG:
                print(f"Processing {country}/{ds} (new: {'_new' in fname})")

            # Copy turbines layer
            tur = gpd.read_file(in_gpkg, layer="turbines").to_crs(PROJECT_CRS)
            tur.to_file(out_gpkg, layer="turbines", driver="GPKG")

            # Read sample hulls (convexhulls)
            sample_hulls = gpd.read_file(in_gpkg, layer=SAMPLE_LAYER).to_crs(PROJECT_CRS)
            sample_hulls["hull_id"] = sample_hulls.index

            if "_new" in fname:
                # 1) Full CDTile splinter and sampling
                tiles = read_raster_bounds(f"data/CDTiles/{country.capitalize()}/")
                combined = process_hulls_and_tiles(sample_hulls, tiles)

                # 2) Generate CDTile-based points
                pts_list = []
                num_tiles = len(tiles)
                for i, t in enumerate(tiles.itertuples(index=False), start=1):
                    print(f"Processing Tile {i} of {num_tiles}", end="", flush=True)
                    match = next(
                        (
                            f
                            for f in os.listdir(f"data/CDTiles/{country.capitalize()}/")
                            if f.startswith(t.tile_name) and f.endswith("_binary.tif")
                        ),
                        None,
                    )
                    if not match:
                        continue
                    mr, tf, cr = mask_raster_with_polygon_and_recode(
                        t.tile_name, os.path.join(f"data/CDTiles/{country.capitalize()}/", match), combined
                    )
                    pts = raster_to_point_gdf(mr, tf, cr).to_crs(PROJECT_CRS)
                    pts["channel"] = t.channel
                    pts_list.append(pts)
                all_pts = gpd.GeoDataFrame(pd.concat(pts_list, ignore_index=True), crs=PROJECT_CRS)

                # 3) Sample CLC rasters on those points
                for key, info in CLC_FILES.items():
                    all_pts = sample_raster_with_points(f"data/CLC/{info['file']}", all_pts, key)

                # 4) For each actual layer, clip points to that layer and compute per-hull stats
                for lyr in LAYERS:
                    hs = gpd.read_file(in_gpkg, layer=lyr).to_crs(PROJECT_CRS)
                    hs["hull_id"] = hs.index
                    joined = gpd.sjoin(all_pts, hs[["hull_id", "geometry"]], how="left", predicate="within")
                    all_stats = pd.DataFrame()
                    for key in CLC_FILES:
                        df = compute_statistics_by_hull_id(joined, key)
                        all_stats = df if all_stats.empty else all_stats.merge(df, on="hull_id")

                    chan_df = (
                        joined.groupby("hull_id")["channel"]
                        .agg(lambda x: x.mode()[0] if not x.mode().empty else "")
                        .reset_index()
                    )

                    out = hs.merge(all_stats, on="hull_id", how="left").fillna(0)
                    out = out.merge(chan_df, on="hull_id", how="left")
                    out["channel"] = out["channel"].fillna("")
                    out["area"] = out.geometry.area.astype(int)
                    out = clean_and_sort(out)
                    out.to_file(out_gpkg, layer=lyr, driver="GPKG")

            else:
                # Direct CLC-only sampling per hull polygon
                for lyr in LAYERS:
                    hs = gpd.read_file(in_gpkg, layer=lyr).to_crs(PROJECT_CRS)
                    hs["hull_id"] = hs.index
                    stats = sample_only_clc(hs, CLC_FILES)
                    out = hs.merge(stats, on="hull_id", how="left").fillna(0).infer_objects(copy=False)
                    out["channel"] = ""
                    out["area"] = out.geometry.area.astype(int)
                    out = clean_and_sort(out)
                    out.to_file(out_gpkg, layer=lyr, driver="GPKG")


if __name__ == "__main__":
    tyro.cli(main)
