#!/usr/bin/env python3
"""
process_turbines.py

Process and merge DTU, OSM and 'OWN' wind‑turbine shapefiles into GeoPackages.

Usage:
    python process_turbines.py [OPTIONS]

This script performs the following steps:
    1. Read DTU and OSM turbine shapefiles
    2. Reproject to EPSG:3035 and remove duplicates
    3. Rename and filter DTU/OSM attributes
    4. Process country-specific "new" and "all" shapefiles:
       - Concatenate, reproject
       - Spatial join nearest DTU/OSM attributes
       - Build unified fields and remove duplicates
    5. Write complete_new.gpkg and complete_all.gpkg
    6. Identify "old" turbines not in "new" within 250m → complete_old.gpkg
    7. Identify "offshore" turbines using contracted OCEAN mask and NUTS exclusion → complete_offshore.gpkg
    8. Identify "other-country" turbines via NUTS buffer → complete_other.gpkg
    9. Generate a summary CSV of feature counts and duplicates
"""

import os
from dataclasses import dataclass
from glob import glob

import geopandas as gpd
import pandas as pd
import tyro


def verbose(msg: str):
    print(f"[INFO] {msg}")


@dataclass
class Config:
    output_dir: str = "data/00_processedturbines"
    turbines_dir: str = "data/TURBINES"
    dtu_osm_dir: str = "data/DTU_OSM"
    crs_input_default: int = 4326
    crs_target: int = 3035
    join_distance: float = 1500.0
    old_distance_threshold: float = 250.0
    dataset_types: list[str] = None

    def __post_init__(self):
        if self.dataset_types is None:
            self.dataset_types = ["new", "all"]


RENAME_MAP_DTU = {
    "Hub_height": "DTU_hubheight",
    "P_rated_kW": "DTU_rpower",
    "Diameter_m": "DTU_diameter",
    "Commission": "DTU_cdate",
}

RENAME_MAP_OSM = {
    "hubheight": "OSM_hubheight",
    "generator": "OSM_model",
    "ratedKW": "OSM_rpower",
    "rotordia": "OSM_diameter",
}


def remove_duplicate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    verbose(f"remove_duplicate_geometries: input {len(gdf)} features")
    if gdf.empty:
        verbose("Input is empty, returning as-is.")
        return gdf
    temp = gdf.copy()
    temp["_wkt"] = temp.geometry.apply(lambda geom: geom.wkt if geom is not None else None)
    deduped = temp.drop_duplicates(subset=["_wkt"]).drop(columns=["_wkt"])
    verbose(f"Removed {len(gdf) - len(deduped)} duplicates → {len(deduped)} features")
    return deduped.reset_index(drop=True)


def remove_close_geometries(gdf: gpd.GeoDataFrame, distance: float = 50.0) -> gpd.GeoDataFrame:
    verbose(f"remove_close_geometries: removing features within {distance} meters")
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["keep"] = True
    # iterate over index/geometry pairs
    for idx, geom in zip(gdf.index, gdf.geometry):
        if not gdf.at[idx, "keep"]:
            continue
        dists = gdf.geometry.distance(geom)
        close = dists[(dists > 0) & (dists <= distance)].index
        gdf.loc[close, "keep"] = False
    result = gdf[gdf["keep"]].drop(columns=["keep"]).reset_index(drop=True)
    verbose(f"  → Removed {len(gdf) - len(result)} close features")
    return result


def process_DTUturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    verbose(f"process_DTUturbines: input columns {list(gdf.columns)}")
    df = gdf.rename(columns=RENAME_MAP_DTU)
    keep = list(RENAME_MAP_DTU.values()) + [df.geometry.name]
    existing = [c for c in keep if c in df.columns]
    verbose(f"Keeping columns: {existing}")
    return df[existing]


def process_OSMturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    verbose(f"process_OSMturbines: input columns {list(gdf.columns)}")
    df = gdf.rename(columns=RENAME_MAP_OSM)
    keep = list(RENAME_MAP_OSM.values()) + [df.geometry.name]
    existing = [c for c in keep if c in df.columns]
    verbose(f"Keeping columns: {existing}")
    return df[existing]


def process_turbines(config: Config):
    verbose("Starting full turbine processing pipeline")
    os.makedirs(config.output_dir, exist_ok=True)
    stats = []

    # === DTU & OSM Reference ===
    verbose("Loading DTU reference data")
    dtu_raw = gpd.read_file(os.path.join(config.dtu_osm_dir, "DTUturbines.shp")).to_crs(epsg=config.crs_target)
    dtu_named = process_DTUturbines(dtu_raw)
    dtu_proc = remove_duplicate_geometries(dtu_named)
    stats.append({
        "dataset": "DTU",
        "count_before_dedup": len(dtu_named),
        "count_after_dedup": len(dtu_proc),
        "duplicates_removed": len(dtu_named) - len(dtu_proc),
        "rpower_count": dtu_proc["DTU_rpower"].notna().sum(),
    })
    dtu_proc.to_file(os.path.join(config.output_dir, "DTUturbines.gpkg"), driver="GPKG")

    verbose("Loading OSM reference data")
    osm_raw = gpd.read_file(os.path.join(config.dtu_osm_dir, "OSMturbines.shp")).to_crs(epsg=config.crs_target)
    osm_named = process_OSMturbines(osm_raw)
    osm_proc = remove_duplicate_geometries(osm_named)
    stats.append({
        "dataset": "OSM",
        "count_before_dedup": len(osm_named),
        "count_after_dedup": len(osm_proc),
        "duplicates_removed": len(osm_named) - len(osm_proc),
        "rpower_count": osm_proc["OSM_rpower"].notna().sum(),
    })
    osm_proc.to_file(os.path.join(config.output_dir, "OSMturbines.gpkg"), driver="GPKG")

    # === New / All Datasets ===
    for ds in config.dataset_types:
        verbose(f"Processing dataset: {ds}")
        files = glob(os.path.join(config.turbines_dir, f"**/*{ds}.shp"), recursive=True)
        gdfs = []
        for fpath in files:
            tmp = gpd.read_file(fpath)
            if tmp.crs is None:
                tmp.set_crs(epsg=config.crs_input_default, inplace=True)
            gdfs.append(tmp.to_crs(epsg=config.crs_target))
        combined = (
            gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            if gdfs
            else gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{config.crs_target}")
        )
        combined = combined.loc[
            :, combined.columns.intersection(["ct_code", "tile_best", "tile", combined.geometry.name])
        ]
        if not combined.empty:
            combined = gpd.sjoin_nearest(combined, dtu_proc, how="left", max_distance=config.join_distance).drop(
                columns=["index_right"], errors="ignore"
            )
            combined = gpd.sjoin_nearest(combined, osm_proc, how="left", max_distance=config.join_distance).drop(
                columns=["index_right"], errors="ignore"
            )
            combined["hubheight"] = combined["DTU_hubheight"].fillna(combined["OSM_hubheight"])
            combined["diameter"] = combined["DTU_diameter"].fillna(combined["OSM_diameter"])
            combined["rpower"] = combined["DTU_rpower"].fillna(combined["OSM_rpower"])
            combined["model"] = combined["OSM_model"]
            combined["cdate"] = combined["DTU_cdate"]
        stats.append({
            "dataset": f"complete_{ds}",
            "count_before_dedup": len(combined),
            "count_after_dedup": len(remove_duplicate_geometries(combined)),
            "duplicates_removed": 0,
            "rpower_count": combined["rpower"].notna().sum(),
        })
        combined = remove_duplicate_geometries(combined)
        combined.to_file(os.path.join(config.output_dir, f"complete_{ds}.gpkg"), driver="GPKG")

    # === Old Turbines ===
    verbose("Identifying old turbines")
    all_pth = os.path.join(config.output_dir, "complete_all.gpkg")
    new_pth = os.path.join(config.output_dir, "complete_new.gpkg")
    if os.path.exists(all_pth) and os.path.exists(new_pth):
        g_all = gpd.read_file(all_pth)
        g_new = gpd.read_file(new_pth)
        joined = gpd.sjoin_nearest(
            g_all, g_new[[g_new.geometry.name]], how="left", max_distance=config.old_distance_threshold
        )
        idx_close = joined[~joined["index_right"].isna()].index
        g_old_pre = joined.loc[~joined.index.isin(idx_close)].drop(columns=["index_right"], errors="ignore")
        g_old_pre["hubheight"] = g_old_pre["DTU_hubheight"].fillna(g_old_pre["OSM_hubheight"])
        g_old_pre["diameter"] = g_old_pre["DTU_diameter"].fillna(g_old_pre["OSM_diameter"])
        g_old_pre["rpower"] = g_old_pre["DTU_rpower"].fillna(g_old_pre["OSM_rpower"])
        g_old_pre["model"] = g_old_pre["OSM_model"]
        g_old_pre["cdate"] = g_old_pre["DTU_cdate"]
        stats.append({
            "dataset": "complete_old",
            "count_before_dedup": len(g_old_pre),
            "count_after_dedup": len(remove_duplicate_geometries(g_old_pre)),
            "duplicates_removed": 0,
            "rpower_count": g_old_pre["rpower"].notna().sum(),
        })
        g_old = remove_duplicate_geometries(g_old_pre)
        g_old.to_file(os.path.join(config.output_dir, "complete_old.gpkg"), driver="GPKG")
    else:
        verbose("Skipping old turbines (missing complete_all or complete_new)")

    # === Offshore Turbines ===
    verbose("Identifying offshore turbines")
    ocean = gpd.read_file("data/REGIONS/OCEAN.gpkg").to_crs(epsg=config.crs_target)
    nuts = gpd.read_file("data/REGIONS/NUTS.gpkg").to_crs(epsg=config.crs_target)

    # Merge DTU + OSM reference
    ref = pd.concat([dtu_proc, osm_proc], ignore_index=True)
    ref["hubheight"] = ref["DTU_hubheight"].fillna(ref["OSM_hubheight"])
    ref["diameter"] = ref["DTU_diameter"].fillna(ref["OSM_diameter"])
    ref["rpower"] = ref["DTU_rpower"].fillna(ref["OSM_rpower"])
    ref["model"] = ref["OSM_model"]
    ref["cdate"] = ref["DTU_cdate"]
    ref = gpd.GeoDataFrame(ref, crs=config.crs_target)

    # Step 1: include only turbines within ocean polygons contracted by 100m
    verbose("Step 1: Filter turbines within contracted OCEAN (-100m)")
    ocean_union = ocean.geometry.union_all()
    ocean_contracted = ocean_union.buffer(-100)
    offshore_step1 = ref[ref.geometry.within(ocean_contracted)].copy()
    verbose(f"  → Retained {len(offshore_step1)} turbines in contracted OCEAN")

    # Step 2: remove turbines within dissolved NUTS buffered by 100m
    verbose("Step 2: Exclude turbines in NUTS buffer (+100m)")
    nuts_union = nuts.geometry.union_all()
    nuts_buffered = nuts_union.buffer(100)
    offshore_final = offshore_step1[~offshore_step1.geometry.intersects(nuts_buffered)].copy()
    verbose(f"  → Final offshore turbines: {len(offshore_final)}")

    stats.append({
        "dataset": "complete_offshore",
        "count_before_dedup": len(ref),
        "count_after_dedup": len(remove_duplicate_geometries(offshore_final)),
        "duplicates_removed": 0,
        "rpower_count": offshore_final["rpower"].notna().sum(),
    })

    g_off = remove_duplicate_geometries(offshore_final)
    g_off = remove_close_geometries(g_off, distance=50)
    g_off.to_file(os.path.join(config.output_dir, "complete_offshore.gpkg"), driver="GPKG")

    # === Other-country Turbines ===
    verbose("Identifying other-country turbines")
    nuts3 = gpd.read_file("data/REGIONS/NUTS.gpkg").to_crs(epsg=config.crs_target)
    nuts3 = nuts3.dissolve(by="ICC").reset_index()

    existing = []
    for path in [all_pth, new_pth, os.path.join(config.output_dir, "complete_old.gpkg")]:
        if os.path.exists(path):
            existing.append(gpd.read_file(path))

    if existing:
        base = pd.concat(existing, ignore_index=True)
        join = gpd.sjoin(nuts3, base.set_geometry(base.geometry.name), how="left", predicate="intersects")
        nuts_other = join[join["index_right"].isna()].drop(columns=["index_right"])
    else:
        nuts_other = nuts3.copy()

    other_union = nuts_other.geometry.union_all()
    other_buffered = other_union.buffer(100)

    combined_ref = pd.concat([dtu_proc, osm_proc], ignore_index=True)
    combined_ref["hubheight"] = combined_ref["DTU_hubheight"].fillna(combined_ref["OSM_hubheight"])
    combined_ref["diameter"] = combined_ref["DTU_diameter"].fillna(combined_ref["OSM_diameter"])
    combined_ref["rpower"] = combined_ref["DTU_rpower"].fillna(combined_ref["OSM_rpower"])
    combined_ref["model"] = combined_ref["OSM_model"]
    combined_ref["cdate"] = combined_ref["DTU_cdate"]
    combined_ref = gpd.GeoDataFrame(combined_ref, crs=config.crs_target)

    other_pre = combined_ref[combined_ref.geometry.within(other_buffered)].copy()

    stats.append({
        "dataset": "complete_other",
        "count_before_dedup": len(combined_ref),
        "count_after_dedup": len(remove_duplicate_geometries(other_pre)),
        "duplicates_removed": 0,
        "rpower_count": other_pre["rpower"].notna().sum(),
    })

    g_other = remove_duplicate_geometries(other_pre)
    g_other = remove_close_geometries(g_other, distance=50)
    g_other.to_file(os.path.join(config.output_dir, "complete_other.gpkg"), driver="GPKG")

    # === Write CSV Summary ===
    stats_df = pd.DataFrame(stats)
    csv_pth = os.path.join(config.output_dir, "processing_stats.csv")
    stats_df.to_csv(csv_pth, index=False)
    verbose(f"Saved processing stats CSV: {csv_pth}")

    verbose("Processing complete 🚀")


def main():
    config = tyro.cli(Config)
    verbose("Configuration:")
    for k, v in vars(config).items():
        verbose(f"  {k}: {v}")
    input("Press ENTER to start processing or Ctrl+C to cancel…")
    process_turbines(config)


if __name__ == "__main__":
    main()
