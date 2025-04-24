#!/usr/bin/env python3
"""
Aggregate CLC-based statistics for turbine hull layers across countries and datasets.

This script:
- Reads processed GeoPackages containing CLC-enriched polygon layers.
- Loads CLC lookup tables from an Excel file.
- Reads detailed turbine attributes from a point dataset.
- Joins nearest point attributes to turbines (within MAX_DISTANCE).
- Summarizes point attributes within polygons (mean for numeric, first for categorical), or nearest if none inside.
- Computes CLC class statistics per polygon layer.
- Exports both GeoPackage layers (with per-feature VAL/CD sums, scaled ×100, and turbine attributes) and per-country Excel reports.

Usage:
    python process_clc_stats.py [--countries austria,france]
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_CRS = "EPSG:3035"
INPUT_DIR = "data/04_clc"
OUTPUT_DIR = "data/05_stat"
CLC_LOOKUP = "data/CLC/CLC.xlsx"
DATASETS = ["new", "old", "all"]
POLY_LAYERS = ["bufferpolys", "concavehulls", "convexhulls"]
CLC_TYPES = ["clc100", "clc10"]
TURB_POINT_GPKG = "data/00_processedturbines/complete_all.gpkg"
MAX_DISTANCE = 5000  # meters
RESULT_FIELDS = ["area", "carea", "careap", "ashare"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CLC statistics for turbine hulls per country and dataset.")
    parser.add_argument(
        "--countries",
        type=str,
        help="Comma-separated list of country folder names to process.",
    )
    return parser.parse_args()


def get_countries(base_dir: str, override: Optional[List[str]] = None) -> List[str]:
    if override:
        return [c.strip() for c in override]
    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError:
        return []
    return sorted([d for d in entries if os.path.isdir(os.path.join(base_dir, d))])


def load_clc_lookup(path: str) -> Dict[str, pd.DataFrame]:
    lookup: Dict[str, pd.DataFrame] = {}
    for clc in CLC_TYPES:
        try:
            df = pd.read_excel(path, sheet_name=clc)
            if "Category" in df.columns and "CatName" in df.columns:
                df["CatName"] = df.groupby("Category")["CatName"].ffill()
            lookup[clc] = df
        except Exception:
            lookup[clc] = pd.DataFrame()
    return lookup


def compute_stats_for_layer(poly_gdf: gpd.GeoDataFrame, clc_type: str, lookup_df: pd.DataFrame) -> pd.DataFrame:
    val_pattern = re.compile(rf"^{clc_type}_VAL_(\d+)$")
    cd_pattern = re.compile(rf"^{clc_type}_CD_(\d+)$")

    val_cols = [c for c in poly_gdf.columns if val_pattern.match(c)]
    cd_cols = [c for c in poly_gdf.columns if cd_pattern.match(c)]

    total_val = poly_gdf[val_cols].sum() if val_cols else pd.Series(dtype=int)
    total_cd = poly_gdf[cd_cols].sum() if cd_cols else pd.Series(dtype=int)

    total_val_sum = int(total_val.sum())
    total_cd_sum = int(total_cd.sum())

    rows = []
    for col in sorted(total_val.index):
        val = int(val_pattern.match(col).group(1))
        area = int(total_val[col])
        carea = int(total_cd.get(col.replace("VAL", "CD"), 0))
        careap = round(carea / area * 100, 2) if area > 0 else 0
        ashare = round(area / total_val_sum * 100, 2) if total_val_sum > 0 else 0
        rows.append({
            "Value": val,
            "area": area,
            "carea": carea,
            "careap": careap,
            "ashare": ashare,
        })

    stats_df = pd.DataFrame(rows).set_index("Value")
    if "Value" in lookup_df:
        vals = lookup_df["Value"].dropna().astype(int).unique()
        stats_df = stats_df.reindex(vals, fill_value=0)

    stats_df[f"{clc_type}_VAL_sum"] = total_val_sum
    stats_df[f"{clc_type}_CD_sum"] = total_cd_sum
    return stats_df


def process_country(
    country: str, args: argparse.Namespace, lookup: Dict[str, pd.DataFrame], point_gdf: gpd.GeoDataFrame
) -> None:
    in_dir = os.path.join(INPUT_DIR, country)
    out_gpkg_dir = os.path.join(OUTPUT_DIR, "gpkg")
    out_xls_dir = os.path.join(OUTPUT_DIR, "xls")
    os.makedirs(out_gpkg_dir, exist_ok=True)
    os.makedirs(out_xls_dir, exist_ok=True)

    for ds in DATASETS:
        src_path = os.path.join(in_dir, f"{country}_{ds}.gpkg")
        if not os.path.exists(src_path):
            continue

        # Load turbines and polygons
        gdf_turb = gpd.read_file(src_path, layer="turbines").to_crs(PROJECT_CRS)
        polys = {ly: gpd.read_file(src_path, layer=ly).to_crs(PROJECT_CRS) for ly in POLY_LAYERS}

        # Map point attributes to turbines (point-to-point)
        attrs = ["hubheight", "diameter", "rpower", "model", "cdate"]
        turb_with_attrs = gpd.sjoin_nearest(
            gdf_turb,
            point_gdf[attrs + ["geometry"]],
            how="left",
            max_distance=MAX_DISTANCE,
            distance_col="_dist",
            lsuffix="",
            rsuffix="_pt",
        )
        for a in attrs:
            col_pt = f"{a}_pt"
            if col_pt in turb_with_attrs:
                gdf_turb[a] = turb_with_attrs[col_pt]

        # Summarize point attributes within polygons
        for ly, gdf in polys.items():
            joined = gpd.sjoin(point_gdf, gdf[["geometry"]], how="inner", predicate="within")
            grp = joined.groupby(joined.index_right)
            means = grp[["hubheight", "diameter", "rpower"]].mean()
            firsts = grp[["model", "cdate"]].first()
            gdf = gdf.join(means).join(firsts)
            empty = gdf[means.columns].isna().all(axis=1)
            if empty.any():
                near = gpd.sjoin_nearest(
                    gdf[empty],
                    point_gdf[attrs + ["geometry"]],
                    how="left",
                    max_distance=MAX_DISTANCE,
                    distance_col="_dist2",
                    lsuffix="",
                    rsuffix="_pt",
                )
                for a in attrs:
                    col_pt = f"{a}_pt"
                    if col_pt in near:
                        gdf.loc[empty, a] = near[col_pt].values
            polys[ly] = gdf

        # Add per-feature VAL/CD sums and scale
        for ly, gdf in polys.items():
            for clc in CLC_TYPES:
                cd_cols = [c for c in gdf.columns if c.startswith(f"{clc}_CD_")]
                val_cols = [c for c in gdf.columns if c.startswith(f"{clc}_VAL_")]
                gdf[f"{clc}_CD_sum"] = gdf[cd_cols].sum(axis=1).astype(int) if cd_cols else 0
                gdf[f"{clc}_VAL_sum"] = gdf[val_cols].sum(axis=1).astype(int) if val_cols else 0
                for col in [*cd_cols, *val_cols, f"{clc}_CD_sum", f"{clc}_VAL_sum"]:
                    gdf[col] = gdf[col] * 100
            polys[ly] = gdf

        # Write GeoPackage
        out_gpkg = os.path.join(out_gpkg_dir, f"{country}_{ds}.gpkg")
        if os.path.exists(out_gpkg):
            os.remove(out_gpkg)
        gdf_turb.to_file(out_gpkg, layer="turbines", driver="GPKG")
        for ly in POLY_LAYERS:
            polys[ly].to_file(out_gpkg, layer=ly, driver="GPKG", mode="a")

        # Write Excel report
        xls_path = os.path.join(out_xls_dir, f"{country}_{ds}.xlsx")
        with pd.ExcelWriter(xls_path) as writer:
            for clc in CLC_TYPES:
                lookup_df = lookup.get(clc, pd.DataFrame())
                if lookup_df.empty:
                    continue
                sheet_df = None
                for ly in POLY_LAYERS:
                    stats = compute_stats_for_layer(polys[ly], clc, lookup_df)
                    stats.columns = [f"{ly}_{c}" for c in stats.columns]
                    sheet_df = stats if sheet_df is None else sheet_df.join(stats)
                sheet_df.to_excel(writer, sheet_name=clc)


def main():
    args = parse_args()
    override = args.countries.split(",") if args.countries else None
    countries = get_countries(INPUT_DIR, override)
    if not countries:
        print(f"No countries found in '{INPUT_DIR}'. Exiting.")
        return

    lookup = load_clc_lookup(CLC_LOOKUP)
    point_gdf = gpd.read_file(TURB_POINT_GPKG).to_crs(PROJECT_CRS)
    for country in tqdm(countries, desc="Countries"):
        process_country(country, args, lookup, point_gdf)


if __name__ == "__main__":
    main()
