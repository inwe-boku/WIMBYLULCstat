#!/usr/bin/env python3
"""
process_turbines.py

Process and merge DTU, OSM and 'OWN' wind‐turbine shapefiles into GeoPackages.

Usage:
    python process_turbines.py

This script:
  1. Reads DTU and OSM turbine shapefiles.
  2. Projects them to EPSG:3035.
  3. Renames and filters relevant attributes.
  4. Reads countrywise "new" and "all" turbine shapefiles, reprojects and concatenates.
  5. Spatially joins nearest DTU/OSM attributes (within 1500 m).
  6. Builds unified fields (hubheight, diameter, rpower, model, cdate).
  7. Writes out:
       - DTUturbines.gpkg
       - OSMturbines.gpkg
       - complete_new.gpkg
       - complete_all.gpkg
  8. Identifies "old" turbines (in complete_all but not in complete_new within 250 m)
     and writes complete_old.gpkg if applicable.

"""

import os
from glob import glob

import geopandas as gpd
import pandas as pd

# ==== CONFIGURATION ====

# Input / output directories
OUTPUT_DIR = "data/00_processedturbines"
TURBINES_DIR = "data/TURBINES"
DTU_OSM_DIR = "data/DTU_OSM"

# CRS definitions
CRS_INPUT_DEFAULT = 4326  # for shapefiles with no CRS
CRS_TARGET = 3035  # ETRS89‐LAEA Europe

# Spatial join distances (in meters)
JOIN_DISTANCE = 1500  # for matching DTU/OSM to local data
OLD_DISTANCE_THRESHOLD = 250  # for identifying "old" turbines

# Dataset types to process
DATASET_TYPES = ["new", "all"]

# Rename maps for source datasets
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


def process_DTUturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Process a DTU turbines GeoDataFrame.

    Keeps only the geometry column and selected DTU attributes, renaming them to:
      - Hub_height → DTU_hubheight
      - P_rated_kW → DTU_rpower
      - Diameter_m → DTU_diameter
      - Commission → DTU_cdate

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame from DTUturbines.shp. Must contain the keys of RENAME_MAP_DTU.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame with only the renamed DTU columns and the geometry.
    """
    # Apply global rename map
    gdf = gdf.rename(columns=RENAME_MAP_DTU)

    # Keep only renamed columns + geometry
    keep_cols = list(RENAME_MAP_DTU.values()) + [gdf.geometry.name]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    return gdf[existing_cols]


def process_OSMturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Process an OSM turbines GeoDataFrame.

    Keeps only the geometry column and selected OSM attributes, renaming them to:
      - hubheight → OSM_hubheight
      - generator → OSM_model
      - ratedKW   → OSM_rpower
      - rotordia  → OSM_diameter

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame from OSMturbines.shp. Must contain the keys of RENAME_MAP_OSM.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame with only the renamed OSM columns and the geometry.
    """
    # Apply global rename map
    gdf = gdf.rename(columns=RENAME_MAP_OSM)

    # Keep only renamed columns + geometry
    keep_cols = list(RENAME_MAP_OSM.values()) + [gdf.geometry.name]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    return gdf[existing_cols]


def main():
    """
    Main entry point for processing turbine datasets.

    Workflow:
      1. Create OUTPUT_DIR.
      2. Read & reproject DTU/OSM source shapefiles; process and save them.
      3. For each type in DATASET_TYPES:
         a. Glob shapefiles under TURBINES_DIR/**/*{type}.shp.
         b. Read, set CRS (CRS_INPUT_DEFAULT if missing), project to CRS_TARGET.
         c. Concatenate or create empty GeoDataFrame.
         d. Keep only ['ct_code', 'tile_best', 'tile', geometry].
         e. Spatially join DTU and OSM processed data within JOIN_DISTANCE.
         f. Build unified fields: hubheight, diameter, rpower, model, cdate.
         g. Write complete_{type}.gpkg to OUTPUT_DIR.
      4. If both complete_all and complete_new exist:
         a. Identify "old" turbines in complete_all not in complete_new within
            OLD_DISTANCE_THRESHOLD.
         b. Write complete_old.gpkg and print confirmation.
      5. Otherwise, print a warning.
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read and reproject source DTU/OSM shapefiles
    dtuturbines = gpd.read_file(os.path.join(DTU_OSM_DIR, "DTUturbines.shp"))
    osmturbines = gpd.read_file(os.path.join(DTU_OSM_DIR, "OSMturbines.shp"))
    dtuturbines.to_crs(epsg=CRS_TARGET, inplace=True)
    osmturbines.to_crs(epsg=CRS_TARGET, inplace=True)

    # Process and save DTU/OSM
    dtuturbines_processed = process_DTUturbines(dtuturbines)
    osmturbines_processed = process_OSMturbines(osmturbines)
    dtuturbines_processed.to_file(os.path.join(OUTPUT_DIR, "DTUturbines.gpkg"), driver="GPKG")
    osmturbines_processed.to_file(os.path.join(OUTPUT_DIR, "OSMturbines.gpkg"), driver="GPKG")

    # Process each dataset type
    for dataset_type in DATASET_TYPES:
        pattern = os.path.join(TURBINES_DIR, f"**/*{dataset_type}.shp")
        shp_files = glob(pattern, recursive=True)

        # Read and reproject local shapefiles
        gdf_list = []
        for f in shp_files:
            tmp = gpd.read_file(f)
            if tmp.crs is None:
                tmp.set_crs(epsg=CRS_INPUT_DEFAULT, inplace=True)
            tmp.to_crs(epsg=CRS_TARGET, inplace=True)
            gdf_list.append(tmp)

        # Concatenate or initialize empty GeoDataFrame
        if gdf_list:
            combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        else:
            combined_gdf = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{CRS_TARGET}")

        # Keep only key fields + geometry
        geom_col = combined_gdf.geometry.name
        combined_gdf = combined_gdf.loc[
            :, combined_gdf.columns.intersection(["ct_code", "tile_best", "tile", geom_col])
        ]

        # Spatial join: DTU attributes
        if not combined_gdf.empty:
            combined_gdf = gpd.sjoin_nearest(
                combined_gdf,
                dtuturbines_processed,
                how="left",
                max_distance=JOIN_DISTANCE,
            )
            combined_gdf.drop(columns=["index_right"], errors="ignore", inplace=True)

        # Spatial join: OSM attributes
        if not combined_gdf.empty:
            combined_gdf = gpd.sjoin_nearest(
                combined_gdf,
                osmturbines_processed,
                how="left",
                max_distance=JOIN_DISTANCE,
            )
            combined_gdf.drop(columns=["index_right"], errors="ignore", inplace=True)

        # Build unified fields
        if not combined_gdf.empty:
            combined_gdf["hubheight"] = combined_gdf["OSM_hubheight"]
            mask = ~combined_gdf["DTU_hubheight"].isna()
            combined_gdf.loc[mask, "hubheight"] = combined_gdf.loc[mask, "DTU_hubheight"]

            combined_gdf["diameter"] = combined_gdf["OSM_diameter"]
            mask = ~combined_gdf["DTU_diameter"].isna()
            combined_gdf.loc[mask, "diameter"] = combined_gdf.loc[mask, "DTU_diameter"]

            combined_gdf["rpower"] = combined_gdf["OSM_rpower"]
            mask = ~combined_gdf["DTU_rpower"].isna()
            combined_gdf.loc[mask, "rpower"] = combined_gdf.loc[mask, "DTU_rpower"]

            combined_gdf["model"] = combined_gdf["OSM_model"]
            combined_gdf["cdate"] = combined_gdf["DTU_cdate"]

        # Write output GeoPackage
        out_file = os.path.join(OUTPUT_DIR, f"complete_{dataset_type}.gpkg")
        combined_gdf.to_file(out_file, driver="GPKG")

    # Identify and save "old" turbines
    complete_all_path = os.path.join(OUTPUT_DIR, "complete_all.gpkg")
    complete_new_path = os.path.join(OUTPUT_DIR, "complete_new.gpkg")

    if os.path.exists(complete_all_path) and os.path.exists(complete_new_path):
        gdf_all = gpd.read_file(complete_all_path)
        gdf_new = gpd.read_file(complete_new_path)

        if not gdf_all.empty and not gdf_new.empty:
            joined = gpd.sjoin_nearest(
                gdf_all,
                gdf_new[[gdf_new.geometry.name]],
                how="left",
                max_distance=OLD_DISTANCE_THRESHOLD,
            )
            to_remove = joined[~joined["index_right"].isna()].index
            gdf_old = joined.loc[~joined.index.isin(to_remove)]
            gdf_old.drop(columns=["index_right"], errors="ignore", inplace=True)
        else:
            gdf_old = gdf_all

        out_old = os.path.join(OUTPUT_DIR, "complete_old.gpkg")
        gdf_old.to_file(out_old, driver="GPKG")
        print(f"Created an additional dataset: {out_old}")
    else:
        print("WARNING: Could not create 'complete_old.gpkg' because one or both:")
        print("  - complete_all.gpkg")
        print("  - complete_new.gpkg")
        print("do not exist.")


if __name__ == "__main__":
    main()
