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
    4. Merge DTU & OSM references into a single GeoDataFrame
    5. Process country-specific "new" and "all" shapefiles:
       - Concatenate, reproject
       - Spatial join nearest from ref_merge to attach standardized fields
       - Remove duplicates
    6. Write complete_new.gpkg and complete_all.gpkg
    7. Identify "old" turbines = complete_all − complete_new
       - Copy ref_merge fields via nearest join
       - Remove duplicates → complete_old.gpkg
    8. Offshore & other-country datasets from ref_merge:
       - Offshore: ref_merge points intersecting OCEAN → complete_offshore.gpkg
       - Other: ref_merge points within NUTS3-buffer minus known countries → complete_other.gpkg
    9. Generate summary CSV of feature counts
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


# Rename maps
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
    if gdf.empty:
        return gdf
    temp = gdf.copy()
    temp["_wkt"] = temp.geometry.apply(lambda g: g.wkt if g is not None else None)
    ded = temp.drop_duplicates(subset=["_wkt"]).drop(columns=["_wkt"])
    return ded.reset_index(drop=True)


def process_DTUturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = gdf.rename(columns=RENAME_MAP_DTU)
    cols = list(RENAME_MAP_DTU.values()) + [df.geometry.name]
    return df[[c for c in cols if c in df.columns]]


def process_OSMturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = gdf.rename(columns=RENAME_MAP_OSM)
    cols = list(RENAME_MAP_OSM.values()) + [df.geometry.name]
    return df[[c for c in cols if c in df.columns]]


def process_turbines(config: Config):
    os.makedirs(config.output_dir, exist_ok=True)
    stats = []

    # 1-2) Load & process DTU/OSM
    dtu = gpd.read_file(os.path.join(config.dtu_osm_dir, "DTUturbines.shp")).to_crs(epsg=config.crs_target)
    dtu = process_DTUturbines(dtu)
    dtu = remove_duplicate_geometries(dtu)
    osm = gpd.read_file(os.path.join(config.dtu_osm_dir, "OSMturbines.shp")).to_crs(epsg=config.crs_target)
    osm = process_OSMturbines(osm)
    osm = remove_duplicate_geometries(osm)

    # Stats for raw refs
    stats.append({
        "dataset": "DTU",
        "count_before_dedup": len(dtu),
        "count_after_dedup": len(dtu),
        "rpower_count": dtu["DTU_rpower"].notna().sum(),
    })
    stats.append({
        "dataset": "OSM",
        "count_before_dedup": len(osm),
        "count_after_dedup": len(osm),
        "rpower_count": osm["OSM_rpower"].notna().sum(),
    })

    # 3) Merge references
    ref_merge = pd.concat([dtu, osm], ignore_index=True)
    ref_merge = gpd.GeoDataFrame(ref_merge, crs=dtu.crs)
    ref_merge = remove_duplicate_geometries(ref_merge)
    stats.append({
        "dataset": "DTU_OSM_merge",
        "count_before_dedup": len(ref_merge),
        "count_after_dedup": len(ref_merge),
        "rpower_count": ((ref_merge["DTU_rpower"].notna()) | (ref_merge["OSM_rpower"].notna())).sum(),
    })
    ref_merge.to_file(os.path.join(config.output_dir, "DTU_OSM_merge.gpkg"), driver="GPKG")

    # helper to attach ref fields
    def attach_ref_fields(gdf):
        temp = gpd.sjoin_nearest(gdf, ref_merge, how="left", max_distance=config.join_distance)
        temp["hubheight"] = temp["DTU_hubheight"].fillna(temp["OSM_hubheight"])
        temp["diameter"] = temp["DTU_diameter"].fillna(temp["OSM_diameter"])
        temp["rpower"] = temp["DTU_rpower"].fillna(temp["OSM_rpower"])
        temp["model"] = temp["OSM_model"]
        temp["cdate"] = temp["DTU_cdate"]
        return temp.drop(columns=["index_right"], errors="ignore")

    # 4) complete_new & complete_all
    for ds in config.dataset_types:
        files = glob(os.path.join(config.turbines_dir, f"**/*{ds}.shp"), recursive=True)
        all_gdfs = []
        for f in files:
            g = gpd.read_file(f)
            if g.crs is None:
                g.set_crs(epsg=config.crs_input_default, inplace=True)
            all_gdfs.append(g.to_crs(epsg=config.crs_target))
        gdf = (
            pd.concat(all_gdfs, ignore_index=True)
            if all_gdfs
            else gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{config.crs_target}")
        )
        gdf = gpd.GeoDataFrame(gdf, crs=gdf.crs)
        processed = attach_ref_fields(gdf)
        processed = remove_duplicate_geometries(processed)
        out = os.path.join(config.output_dir, f"complete_{ds}.gpkg")
        processed.to_file(out, driver="GPKG")
        stats.append({
            "dataset": f"complete_{ds}",
            "count_before_dedup": len(gdf),
            "count_after_dedup": len(processed),
            "rpower_count": processed["rpower"].notna().sum(),
        })

    # 5) complete_old = all - new
    all_g = gpd.read_file(os.path.join(config.output_dir, "complete_all.gpkg"))
    new_g = gpd.read_file(os.path.join(config.output_dir, "complete_new.gpkg"))
    diff = gpd.sjoin_nearest(all_g, new_g, how="left", max_distance=config.old_distance_threshold)
    old_pre = diff[diff["index_right"].isna()].drop(columns=["index_right"], errors="ignore")
    old_pre = attach_ref_fields(old_pre)
    old = remove_duplicate_geometries(old_pre)
    old.to_file(os.path.join(config.output_dir, "complete_old.gpkg"), driver="GPKG")
    stats.append({
        "dataset": "complete_old",
        "count_before_dedup": len(old_pre),
        "count_after_dedup": len(old),
        "rpower_count": old["rpower"].notna().sum(),
    })

    # 6) offshore from ref_merge
    ocean = gpd.read_file("data/REGIONS/OCEAN.gpkg").to_crs(epsg=config.crs_target)
    off = ref_merge[ref_merge.geometry.intersects(ocean.geometry.union_all())].copy()
    off = remove_duplicate_geometries(off)
    off.to_file(os.path.join(config.output_dir, "complete_offshore.gpkg"), driver="GPKG")
    stats.append({
        "dataset": "complete_offshore",
        "count_before_dedup": len(ref_merge),
        "count_after_dedup": len(off),
        "rpower_count": off["rpower"].notna().sum(),
    })

    # 7) other-country from ref_merge
    nuts = gpd.read_file("data/REGIONS/NUTS.gpkg").to_crs(epsg=config.crs_target)
    nuts3 = nuts.dissolve(by="ICC").reset_index()
    existing = pd.concat([all_g, new_g, old], ignore_index=True)
    j = gpd.sjoin(nuts3, existing, how="left", predicate="intersects")
    others = nuts3[j["index_right"].isna()].drop(columns=["index_right"], errors="ignore")
    buf = others.geometry.buffer(100)
    union_buf = gpd.GeoSeries(buf).union_all()
    other = ref_merge[ref_merge.geometry.within(union_buf)].copy()
    other = remove_duplicate_geometries(other)
    other.to_file(os.path.join(config.output_dir, "complete_other.gpkg"), driver="GPKG")
    stats.append({
        "dataset": "complete_other",
        "count_before_dedup": len(ref_merge),
        "count_after_dedup": len(other),
        "rpower_count": other["rpower"].notna().sum(),
    })

    # 8) write stats CSV
    pd.DataFrame(stats).to_csv(os.path.join(config.output_dir, "processing_stats.csv"), index=False)


def main():
    config = tyro.cli(Config)
    process_turbines(config)


if __name__ == "__main__":
    main()
