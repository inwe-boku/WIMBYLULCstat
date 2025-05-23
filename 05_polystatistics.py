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
- Exports both GeoPackage layers and per-country Excel reports with detailed and summary sheets.

Usage:
    python process_clc_stats_refactored.py [--countries austria,france]
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

# store skipped datasets for summary
skipped_datasets: List[str] = []


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
        rows.append({"Value": val, "area": area, "carea": carea, "careap": careap, "ashare": ashare})
    stats_df = pd.DataFrame(rows).set_index("Value")
    if "Value" in lookup_df.columns:
        vals = lookup_df["Value"].dropna().astype(int).unique()
        stats_df = stats_df.reindex(vals, fill_value=0)
    stats_df[f"{clc_type}_VAL_sum"] = total_val_sum
    stats_df[f"{clc_type}_CD_sum"] = total_cd_sum
    return stats_df


def process_gpkg(
    country: str,
    ds: str,
    gdf_turb: gpd.GeoDataFrame,
    polys: Dict[str, gpd.GeoDataFrame],
    output_dir: str = OUTPUT_DIR,
) -> None:
    out_gpkg_dir = os.path.join(output_dir, "gpkg")
    os.makedirs(out_gpkg_dir, exist_ok=True)
    out_gpkg = os.path.join(out_gpkg_dir, f"{country}_{ds}.gpkg")
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)
    gdf_turb.to_file(out_gpkg, layer="turbines", driver="GPKG")
    for ly, gdf in polys.items():
        gdf.to_file(out_gpkg, layer=ly, driver="GPKG", mode="a")


def process_xls(
    country: str,
    ds: str,
    polys: Dict[str, gpd.GeoDataFrame],
    lookup: Dict[str, pd.DataFrame],
    gdf_turb: gpd.GeoDataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    out_xls_dir = os.path.join(output_dir, "xls")
    os.makedirs(out_xls_dir, exist_ok=True)
    xls_path = os.path.join(out_xls_dir, f"{country}_{ds}.xlsx")
    import numpy as np

    with pd.ExcelWriter(xls_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({"text_wrap": True, "align": "center"})
        # Detailed CLC sheets
        for clc in CLC_TYPES:
            lookup_df = lookup.get(clc, pd.DataFrame())
            if lookup_df.empty:
                continue
            meta_cols = [c for c in lookup_df.columns if c != "Value"]
            meta = lookup_df.set_index("Value")[meta_cols]
            parts = []
            for ly in POLY_LAYERS:
                stats = compute_stats_for_layer(polys[ly], clc, lookup_df)
                core = stats[["area", "carea", "careap", "ashare"]].copy()
                core.columns = pd.MultiIndex.from_product([[ly], core.columns])
                parts.append(core)
            df_detail = pd.concat(parts, axis=1)
            meta.columns = pd.MultiIndex.from_product([[""], meta.columns])
            df_out = pd.concat([meta, df_detail], axis=1)
            df_out.to_excel(writer, sheet_name=clc, startrow=0)
            ws_clc = writer.sheets[clc]
            ws_clc.set_row(0, 30, header_format)
            ws_clc.set_row(1, 20, header_format)
            col_offset = len(meta_cols) + 1
            for i, ly in enumerate(POLY_LAYERS):
                start = col_offset + i * 4
                end = start + 3
                try:
                    ws_clc.merge_range(0, start, 0, end, ly, header_format)
                except Exception:
                    pass
            for idx, col in enumerate(df_out.columns):
                col_idx = idx + 1
                header = " ".join([str(x) for x in col if x])
                max_len = len(header)
                col_vals = df_out[col].astype(str)
                if not col_vals.empty:
                    max_len = max(max_len, col_vals.map(len).max())
                ws_clc.set_column(col_idx, col_idx, max_len + 2)
        # Summary sheet
        from collections import OrderedDict

        metrics = OrderedDict([
            ("total_turbines", "total turbines"),
            ("num_windparks", "number of windparks"),
            ("avg_turbines_per_concavehulls", "average turbines per windpark"),
            ("avg_hubheight", "average hubheight"),
            ("avg_diameter", "average diameter"),
            ("avg_rpower", "average rpower"),
        ])
        num_turbines = len(gdf_turb)
        num_polys = len(polys["concavehulls"])
        metric_vals = [
            num_turbines,
            num_polys,
            num_turbines / num_polys if num_polys else 0,
            float(gdf_turb["hubheight"].mean()),
            float(gdf_turb["diameter"].mean()),
            float(gdf_turb["rpower"].mean()),
        ]
        clc_vals = []
        for clc in CLC_TYPES:
            for ly in POLY_LAYERS:
                clc_vals.append(int(polys[ly][f"{clc}_VAL_sum"].sum()))
        top = ["Metrics"] * len(metrics) + [clc for clc in CLC_TYPES for _ in POLY_LAYERS]
        bottom = list(metrics.values()) + POLY_LAYERS * len(CLC_TYPES)
        mindex = pd.MultiIndex.from_arrays([top, bottom])
        df_sum = pd.DataFrame([metric_vals + clc_vals], columns=mindex)
        df_sum = df_sum.replace([np.nan, np.inf, -np.inf], 0).round(0).astype(int)
        df_sum.to_excel(writer, sheet_name="summary", startrow=0)
        ws_sum = writer.sheets["summary"]
        ws_sum.set_column(0, 0, None, None, {"hidden": True})
        ws_sum.set_row(0, 30, header_format)
        ws_sum.set_row(1, 30, header_format)
        for idx, col in enumerate(df_sum.columns):
            col_idx = idx + 1
            header = " ".join([str(x) for x in col if x])
            max_len = len(header)
            col_vals = df_sum[col].astype(str)
            if not col_vals.empty:
                max_len = max(max_len, col_vals.map(len).max())
            ws_sum.set_column(col_idx, col_idx, max_len + 2)


def process_country(
    country: str,
    args: argparse.Namespace,
    lookup: Dict[str, pd.DataFrame],
    point_gdf: gpd.GeoDataFrame,
) -> None:
    in_dir = os.path.join(INPUT_DIR, country)
    from tqdm import tqdm as _tqdm

    ds_bar = _tqdm(DATASETS, desc=country, leave=False)
    for ds in ds_bar:
        ds_bar.set_description(f"{country}-{ds}")
        src = os.path.join(in_dir, f"{country}_{ds}.gpkg")
        if not os.path.exists(src):
            skipped_datasets.append(f"{country}-{ds}")
            continue
        try:
            gdf_turb = gpd.read_file(src, layer="turbines").to_crs(PROJECT_CRS)
            polys = {ly: gpd.read_file(src, layer=ly).to_crs(PROJECT_CRS) for ly in POLY_LAYERS}
            attrs = ["hubheight", "diameter", "rpower", "model", "cdate"]
            pt = point_gdf[attrs + ["geometry"]].rename(columns={a: f"{a}_pt" for a in attrs})
            tn = gpd.sjoin_nearest(
                gdf_turb,
                pt,
                how="inner",
                max_distance=MAX_DISTANCE,
                distance_col="_dist",
            )
            orig_cols = list(gdf_turb.columns)
            gdf_turb = tn[orig_cols + [f"{a}_pt" for a in attrs]].rename(columns={f"{a}_pt": a for a in attrs})
            for ly in POLY_LAYERS:
                g = polys[ly]
                joined = gpd.sjoin(point_gdf, g[["geometry"]], how="inner", predicate="within")
                grp = joined.groupby(joined.index_right)
                g = g.join(grp[["hubheight", "diameter", "rpower"]].mean()).join(grp[["model", "cdate"]].first())
                mask = g[["hubheight", "diameter", "rpower"]].isna().all(axis=1)
                if mask.any():
                    near = gpd.sjoin_nearest(
                        g[mask],
                        point_gdf[attrs + ["geometry"]],
                        how="left",
                        max_distance=MAX_DISTANCE,
                        distance_col="_dist2",
                        lsuffix="_poly",
                        rsuffix="_pt",
                    )
                    for a in attrs:
                        if f"{a}_pt" in near.columns:
                            g.loc[mask, a] = near[f"{a}_pt"].values
                polys[ly] = g
            for ly in POLY_LAYERS:
                g = polys[ly]
                for clc in CLC_TYPES:
                    cd_cols = [c for c in g.columns if c.startswith(f"{clc}_CD_")]
                    val_cols = [c for c in g.columns if c.startswith(f"{clc}_VAL_")]
                    g[f"{clc}_CD_sum"] = g[cd_cols].sum(axis=1).astype(int) if cd_cols else 0
                    g[f"{clc}_VAL_sum"] = g[val_cols].sum(axis=1).astype(int) if val_cols else 0
                    for c in (
                        cd_cols
                        + val_cols
                        + [
                            f"{clc}_CD_sum",
                            f"{clc}_VAL_sum",
                        ]
                    ):
                        g[c] *= 100
                    polys[ly] = g
            process_gpkg(country, ds, gdf_turb, polys)
            process_xls(country, ds, polys, lookup, gdf_turb)
        except Exception as err:
            print(f"Error processing country {country}, dataset {ds}: {err}")
            continue


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
    # After processing all, report skipped datasets
    if skipped_datasets:
        print("Skipped the following country-dataset combinations due to missing input:")
        for entry in skipped_datasets:
            print(f" - {entry}")


if __name__ == "__main__":
    main()
