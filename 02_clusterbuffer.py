#!/usr/bin/env python3
"""Process turbine point data to create buffer, convex and concave hulls per cluster.

This script:
- Reads GeoPackage point datasets per country/dataset.
- Performs DBSCAN clustering to group turbines.
- Calculates buffer sizes per cluster.
- Generates buffer polygons, convex hulls, and concave hulls.
- Writes each layer to output GeoPackages.

Configurable via command-line arguments:
    --countries: comma-separated list of country folder names to process.
"""

import argparse
import os

import geopandas as gpd
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.ops import split
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration Variables (override via CLI)
# -----------------------------------------------------------------------------
INPUT_DIR = "data/01_outregions"
OUTPUT_DIR = "data/02_outhulls"
DATASETS = ["new", "all", "old"]
CRS = "epsg:3035"
EPSILON = 1500  # DBSCAN eps distance threshold in CRS units
MIN_SAMPLES = 2  # Minimum samples to form a cluster
BUFFERPERC = 0.5  # Percentage of nearest neighbor distance for buffer
BUFFERSCALE1 = 2 / 3
BUFFERSCALE2 = 1 / 2
SAMPLE_POINTS = 99  # Number of points to sample on buffer boundary
CONCAVE_RATIO = 0.25  # Concave hull ratio parameter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create hull layers around clustered turbine points.")
    parser.add_argument(
        "--countries",
        type=str,
        help="Comma-separated list of country folder names to process (overrides auto-detection).",
    )
    return parser.parse_args()


def get_countries(input_dir, override_list=None):
    """Determine list of countries by scanning input_dir or using override."""
    if override_list:
        return [c.strip() for c in override_list]
    try:
        entries = os.listdir(input_dir)
    except FileNotFoundError:
        return []
    countries = [d for d in entries if os.path.isdir(os.path.join(input_dir, d))]
    return sorted(countries)


def clustercalculation(turbines):
    """Assign DBSCAN clusters and calculate buffer per cluster."""
    coords = [(geom.x, geom.y) for geom in turbines.geometry]
    labels = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit_predict(coords)
    turbines["cluster"] = labels
    cluster_distances = {}
    for cid in np.unique(labels):
        if cid == -1:
            continue
        pts = turbines[turbines.cluster == cid].geometry
        ccoords = np.array([(pt.x, pt.y) for pt in pts])
        dmat = distance.cdist(ccoords, ccoords)
        dmat[dmat < 9] = np.inf
        mins = np.min(dmat, axis=1)
        valids = mins[np.isfinite(mins)]
        buf = int(round(valids.mean() * BUFFERPERC, 0)) if len(valids) > 0 else np.nan
        cluster_distances[cid] = buf
    turbines["buffer"] = turbines.cluster.map(cluster_distances)
    default = int(round(turbines.buffer.dropna().mean(), 0)) if not turbines.buffer.dropna().empty else 0
    turbines["buffer"] = turbines.buffer.fillna(default)
    return turbines


def create_buffer_polygons(points):
    """Generate buffer polygons around points using combined buffer scales."""
    df = points.copy()
    df["geometry"] = points.buffer(points.buffer.values * (BUFFERSCALE1 + BUFFERSCALE2))
    return df


def create_convex_hulls(points):
    """Generate convex hulls per cluster and buffer them."""
    grouped = points.dissolve(by="cluster")
    hulls = grouped.convex_hull.buffer(grouped.buffer.values * (BUFFERSCALE1 + BUFFERSCALE2))
    df = grouped.copy()
    df["geometry"] = hulls
    return df


def sample_buffer_boundaries(points, scale, n_points):
    """Sample n_points along the buffer boundary of each point."""
    records = []
    for _, row in points.iterrows():
        buf_geom = row.geometry.buffer(row.buffer * scale)
        boundary = buf_geom.boundary
        coords = list(boundary.coords)
        step = max(1, len(coords) // n_points)
        for pt in coords[::step]:
            rec = row.to_dict()
            rec["geometry"] = Point(pt)
            records.append(rec)
    return gpd.GeoDataFrame(records, crs=points.crs)


def create_concave_hulls(points):
    """Generate concave hulls via boundary sampling and concave_hull op."""
    bp = sample_buffer_boundaries(points, BUFFERSCALE1, SAMPLE_POINTS)
    grouped = bp.dissolve(by="cluster")
    hulls = grouped.concave_hull(ratio=CONCAVE_RATIO).buffer(grouped.buffer.values * BUFFERSCALE2)
    df = grouped.copy()
    df["geometry"] = hulls
    return df


def create_hulls(points, hull_type="convex"):
    """Wrapper to generate buffer, convex, or concave hulls."""
    if hull_type == "buffer":
        hulls = create_buffer_polygons(points)
    elif hull_type == "convex":
        hulls = create_convex_hulls(points)
    elif hull_type == "concave":
        hulls = create_concave_hulls(points)
    else:
        raise ValueError("Choose 'buffer', 'convex', or 'concave'")
    hulls.reset_index(inplace=True)
    hulls["hull_id"] = hulls.index + 1
    return hulls


def split_hulls(hulls, split_polygons):
    """Split hulls by polygon boundaries and remove duplicates."""
    split_lines = split_polygons.boundary
    results = []
    for _, hull in hulls.iterrows():
        geoms = [hull.geometry]
        for line in split_lines:
            next_geoms = []
            for g in geoms:
                if g.intersects(line):
                    parts = split(g, line)
                    next_geoms.extend([p for p in parts.geoms if p.is_valid and not p.is_empty])
                else:
                    next_geoms.append(g)
            geoms = next_geoms
        for g in geoms:
            rec = hull.to_dict()
            rec["geometry"] = g
            results.append(rec)
    return gpd.GeoDataFrame(results, crs=hulls.crs).drop_duplicates(subset="geometry")


def assign_tile_best(turbines, polygons):
    """Assign most common turbine.tile_best to each polygon."""
    if "tile_best" not in turbines.columns:
        polygons["tile_best"] = None
        return polygons
    turbines = turbines.to_crs(polygons.crs)
    polygons["tile_best"] = None
    for idx, poly in polygons.iterrows():
        pts = turbines[turbines.geometry.within(poly.geometry)]
        if not pts.empty:
            mode = pts["tile_best"].mode()
            if not mode.empty:
                polygons.at[idx, "tile_best"] = mode[0]
    return polygons


def merge_slivers(hulls, slivers):
    """Merge sliver polygons (<1% area) with adjacent ones."""
    if "hull_id" not in hulls.columns or "hull_id" not in slivers.columns:
        raise ValueError("Missing 'hull_id'")
    hulls["original_area"] = hulls.geometry.area
    sl = slivers.copy()
    sl["split_area"] = sl.geometry.area
    sl = sl.merge(hulls[["hull_id", "original_area"]], on="hull_id", how="left")
    sl["area_percentage"] = sl.split_area / sl.original_area * 100
    updated = []
    processed = set()
    for idx, s in sl.iterrows():
        if idx in processed:
            continue
        if s.area_percentage < 1:
            adj = sl[sl.geometry.touches(s.geometry) & ~sl.index.isin(processed)]
            if not adj.empty:
                first = adj.iloc[0]
                merged_geom = s.geometry.union(first.geometry)
                rec = s.to_dict()
                rec["geometry"] = merged_geom
                updated.append(rec)
                processed.update({idx, first.name})
            else:
                updated.append(s.to_dict())
                processed.add(idx)
        else:
            updated.append(s.to_dict())
            processed.add(idx)
    df = gpd.GeoDataFrame(updated, crs=sl.crs).drop_duplicates(subset="geometry")
    df["split_area"] = df.geometry.area
    df["area_percentage"] = df.split_area / df.original_area * 100
    return df


def main():
    """Entry point: parse args, detect countries, process each hull type."""
    args = parse_args()
    override = args.countries.split(",") if args.countries else None
    countries = get_countries(INPUT_DIR, override)
    if not countries:
        print(f"No countries found in '{INPUT_DIR}'. Exiting.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for country in tqdm(countries, desc="Countries"):
        country_out = os.path.join(OUTPUT_DIR, country)
        os.makedirs(country_out, exist_ok=True)
        for ds in DATASETS:
            fp = os.path.join(INPUT_DIR, country, f"{country}_{ds}.gpkg")
            if not os.path.exists(fp):
                continue
            gdf = gpd.read_file(fp)
            if gdf.empty:
                continue
            gdf = gdf.to_crs(CRS).drop_duplicates(subset="geometry")
            gdf = clustercalculation(gdf)
            gdf.to_file(os.path.join(country_out, f"{country}_{ds}.gpkg"), layer="turbines")
            for htype, layer in [("buffer", "bufferpolys"), ("convex", "convexhulls"), ("concave", "concavehulls")]:
                hulls = create_hulls(gdf, htype)
                hulls.to_file(os.path.join(country_out, f"{country}_{ds}.gpkg"), layer=layer)


if __name__ == "__main__":
    main()
