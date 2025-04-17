#!/usr/bin/env python3
"""
Aggregate CLC-based statistics for turbine hull layers across countries and datasets.

This script:
- Reads processed GeoPackages containing CLC-enriched polygon layers.
- Loads CLC lookup tables from an Excel file.
- Computes summary statistics (area, clipped area, percent coverage, area share) per CLC class.
- Exports both GeoPackage layers and per-country Excel reports.

Usage:
    python process_clc_stats.py [--countries austria,france]
"""

import argparse
import os
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration (override via CLI)
# -----------------------------------------------------------------------------
PROJECT_CRS = "EPSG:3035"
INPUT_DIR = "data/04_clc"
OUTPUT_DIR = "data/05_stat"
CLC_LOOKUP = "data/CLC/CLC.xlsx"
DATASETS = ["new", "all"]
POLY_LAYERS = ["bufferpolys", "concavehulls", "convexhulls"]
CLC_TYPES = ["clc100", "clc10"]
RESULT_FIELDS = ["area", "carea", "careap", "ashare"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with 'countries'.
    """
    parser = argparse.ArgumentParser(description="Compute CLC statistics for turbine hulls per country and dataset.")
    parser.add_argument(
        "--countries",
        type=str,
        help="Comma-separated list of country folder names to process (overrides auto-detection).",
    )
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


def load_clc_lookup(path: str) -> Dict[str, pd.DataFrame]:
    """
    Load CLC lookup tables from an Excel file.

    Args:
        path (str): Excel file with sheets for each CLC type.

    Returns:
        Dict[str, DataFrame]: Mapping of clc_type to lookup DataFrame.
    """
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
    """
    For one polygon layer, compute CLC statistics per class.

    Args:
        poly_gdf (GeoDataFrame): Polygon layer with CLC count/value columns.
        clc_type (str): e.g. 'clc100'.
        lookup_df (DataFrame): Lookup table with 'Value' column.

    Returns:
        DataFrame: Indexed by 'Value' with columns [area, carea, careap, ashare].
    """
    # Identify count and value columns
    cd_cols = [c for c in poly_gdf.columns if c.startswith(f"{clc_type}_CD_")]
    val_cols = [c for c in poly_gdf.columns if c.startswith(f"{clc_type}_VAL_")]
    total_cd = poly_gdf[cd_cols].sum() if cd_cols else pd.Series(dtype=int)
    total_val = poly_gdf[val_cols].sum() if val_cols else pd.Series(dtype=int)
    total_val_sum = total_val.sum()

    rows = []
    for col in sorted(total_val.index):
        val = int(col.rsplit("_", 1)[-1])
        area = int(total_val[col])
        carea = int(total_cd.get(col.replace("VAL", "CD"), 0))
        careap = round(carea / area * 100, 2) if area > 0 else 0
        ashare = round(area / total_val_sum * 100, 2) if total_val_sum > 0 else 0
        rows.append({"Value": val, "area": area, "carea": carea, "careap": careap, "ashare": ashare})

    stats_df = pd.DataFrame(rows).set_index("Value")
    # Ensure all lookup values are present
    if "Value" in lookup_df:
        vals = lookup_df["Value"].dropna().astype(int).unique()
        stats_df = stats_df.reindex(vals, fill_value=0)
    return stats_df


def process_country(country: str, args: argparse.Namespace, lookup: Dict[str, pd.DataFrame]) -> None:
    """
    Process all datasets for a single country: compute and save stats.

    Args:
        country (str): Country folder name.
        args (Namespace): Parsed CLI arguments.
        lookup (dict): CLC lookup tables.
    """
    in_dir = os.path.join(INPUT_DIR, country)
    out_gpkg_dir = os.path.join(OUTPUT_DIR, "gpkg")
    out_xls_dir = os.path.join(OUTPUT_DIR, "xls")
    os.makedirs(out_gpkg_dir, exist_ok=True)
    os.makedirs(out_xls_dir, exist_ok=True)

    for ds in DATASETS:
        src_path = os.path.join(in_dir, f"{country}_{ds}.gpkg")
        if not os.path.exists(src_path):
            continue

        # Read and save turbines layer
        gdf_turb = gpd.read_file(src_path, layer="turbines")
        out_gpkg = os.path.join(out_gpkg_dir, f"{country}_{ds}.gpkg")
        gdf_turb.to_file(out_gpkg, layer="turbines", driver="GPKG")

        # Load polygon layers once
        polys = {ly: gpd.read_file(src_path, layer=ly) for ly in POLY_LAYERS}

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

        # Append polygon layers to output GeoPackage
        for ly in POLY_LAYERS:
            gdf = polys[ly]
            gdf.to_file(out_gpkg, layer=ly, driver="GPKG", mode="a")


def main():
    """
    Entry point: parse args, detect countries, load lookup, and process.
    """
    args = parse_args()
    override = args.countries.split(",") if args.countries else None
    countries = get_countries(INPUT_DIR, override)
    if not countries:
        print(f"No countries found in '{INPUT_DIR}'. Exiting.")
        return

    lookup = load_clc_lookup(CLC_LOOKUP)
    for country in tqdm(countries, desc="Countries"):
        process_country(country, args, lookup)


if __name__ == "__main__":
    main()
