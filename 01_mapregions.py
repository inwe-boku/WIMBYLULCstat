import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import geopandas as gpd
import tyro

# ----------------------------------------------------------------------------
# Configuration Variables
# ----------------------------------------------------------------------------
INPUT_DIR = "data/TURBINES"
PROCESSED_DIR = "data/00_processedturbines"
REGIONS_DIR = "data/REGIONS"
OUTPUT_DIR = "data/01_outregions"
TURBINE_LAYER = "turbines"
SHAPE_DATASETS = ["new", "all"]  # datasets provided as shapefiles
EXTRA_FILES_DEFAULT = "complete_other.gpkg,complete_offshore.gpkg"


@dataclass
class Config:
    """Configuration for turbine processing."""

    input_dir: str = INPUT_DIR
    processed_dir: str = PROCESSED_DIR
    regions_dir: str = REGIONS_DIR
    output_dir: str = OUTPUT_DIR
    extra_files: Optional[str] = EXTRA_FILES_DEFAULT
    max_distance: int = 25000
    old_distance: int = 25
    complete_all_dataset: str = "complete_all.gpkg"
    complete_all_distance: int = 25
    complete_all_fields: List[str] = field(
        default_factory=lambda: ["hubheight", "diameter", "rpower", "model", "cdate"]
    )
    process_countries: bool = True
    process_extras: bool = False
    second_datasets_config: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "NUTS.gpkg": ["NUTS1_CODE", "NUTS2_CODE", "NUTS3_CODE", "NUTS3_NAME"],
            "WWF_Ecos.gpkg": ["WWFEcos_code", "WWFEcos_name", "WWFEcos_realm", "WWFEcos_biome"],
            "EEA_Biogeoregions2016.gpkg": ["EEABGR_code", "EEABGR_sname", "EEABGR_name"],
            "EC_ENSv8.gpkg": ["EnS_code", "EnS_name", "EnZ_code", "EnZ_name"],
        }
    )


def get_countries(input_dir: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    except FileNotFoundError:
        return []


def dedupe_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.drop_duplicates(subset=[gdf.geometry.name]).reset_index(drop=True)
    gdf["fid"] = range(1, len(gdf) + 1)
    return gdf


def enrich_with_regions(gdf: gpd.GeoDataFrame, config: Config) -> gpd.GeoDataFrame:
    result = gdf.copy()
    for fname, fields in config.second_datasets_config.items():
        path = os.path.join(config.regions_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] Region file {fname} not found, skipping.")
            continue
        regions = gpd.read_file(path).to_crs(result.crs)
        missing = [f for f in fields if f not in regions.columns]
        if missing:
            print(f"[WARN] Missing fields {missing} in {fname}, skipping.")
            continue
        regions = regions[[regions.geometry.name] + fields]
        joined = gpd.sjoin(result, regions, how="left", predicate="intersects")
        cols = [c for c in fields if c in joined.columns]
        mask = joined[cols].isna().all(axis=1)
        if mask.any():
            subset = joined.loc[mask].drop(columns=cols + ["index_right"], errors="ignore")
            nearest = gpd.sjoin_nearest(
                subset,
                regions,
                how="left",
                max_distance=config.max_distance,
                distance_col="nearest_distance",
            )
            nearest = nearest.loc[~nearest.index.duplicated(keep="first")]
            for col in cols:
                if col in nearest.columns:
                    joined.loc[nearest.index, col] = nearest[col]
        result = (
            joined.drop(columns={"index_right", "nearest_distance"} & set(joined.columns), errors="ignore")
            .loc[~joined.index.duplicated(keep="first")]
            .reset_index(drop=True)
        )
    return result


def map_complete_all(gdf: gpd.GeoDataFrame, complete_all: gpd.GeoDataFrame, config: Config) -> gpd.GeoDataFrame:
    fields = [f for f in config.complete_all_fields if f in complete_all.columns]
    joined = gpd.sjoin_nearest(
        gdf,
        complete_all[[complete_all.geometry.name] + fields],
        how="left",
        max_distance=config.complete_all_distance,
        distance_col="_dist_complete_all",
    )
    for fld in fields:
        gdf[fld] = joined.get(fld)
    return gdf


def process_file(in_fp: str, out_fp: str, config: Config, complete_all: Optional[gpd.GeoDataFrame] = None) -> None:
    if not os.path.exists(in_fp):
        print(f"[SKIP] {os.path.basename(in_fp)} not found.")
        return
    print(f"[START] {os.path.basename(in_fp)}")
    try:
        gdf = gpd.read_file(in_fp)
    except Exception as e:
        print(f"[ERROR] Reading {in_fp}: {e}")
        return
    gdf = dedupe_index(gdf)
    enriched = enrich_with_regions(gdf, config)
    if complete_all is not None:
        enriched = map_complete_all(enriched, complete_all, config)
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    try:
        enriched.to_file(out_fp, layer=TURBINE_LAYER)
        print(f"[DONE] Wrote {os.path.basename(out_fp)}")
    except Exception as e:
        print(f"[ERROR] Writing {out_fp}: {e}")


def main(config: Config) -> None:
    # load complete_all dataset
    ca_fp = os.path.join(config.processed_dir, config.complete_all_dataset)
    complete_all = None
    if os.path.exists(ca_fp):
        complete_all = gpd.read_file(ca_fp)
    # process per-country shapefiles if enabled
    if config.process_countries:
        countries = get_countries(config.input_dir)
        for country in countries:
            country_dir = os.path.join(config.output_dir, country)
            for ds in SHAPE_DATASETS:
                shp = os.path.join(config.input_dir, country, f"{country}_{ds}.shp")
                out_fp = os.path.join(country_dir, f"{country}_{ds}.gpkg")
                process_file(shp, out_fp, config, complete_all)
            # compute old dataset
            new_fp = os.path.join(country_dir, f"{country}_new.gpkg")
            all_fp = os.path.join(country_dir, f"{country}_all.gpkg")
            if os.path.exists(new_fp) and os.path.exists(all_fp):
                gdf_new = gpd.read_file(new_fp, layer=TURBINE_LAYER)
                gdf_all = gpd.read_file(all_fp, layer=TURBINE_LAYER)
                joined = gpd.sjoin_nearest(
                    gdf_all, gdf_new, how="left", max_distance=config.old_distance, distance_col="dist"
                )
                old = joined[joined["dist"].isna()].drop(columns=["index_right", "dist"], errors="ignore")
                if complete_all is not None:
                    old = map_complete_all(old, complete_all, config)
                out_old = os.path.join(country_dir, f"{country}_old.gpkg")
                old.to_file(out_old, layer=TURBINE_LAYER)
    # process extra files if enabled
    if config.process_extras and config.extra_files:
        extras_dir = os.path.join(config.output_dir, "0_extras")
        for fname in [f.strip() for f in config.extra_files.split(",") if f.strip()]:
            in_fp = os.path.join(config.processed_dir, fname)
            out_fp = os.path.join(extras_dir, f"processed_{fname}")
            process_file(in_fp, out_fp, config, complete_all)
    print("=== Processing complete ===")


if __name__ == "__main__":
    tyro.cli(main)
