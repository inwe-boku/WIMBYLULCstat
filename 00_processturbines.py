import os
from glob import glob

import geopandas as gpd
import pandas as pd


def process_DTUturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only geometry and rename:
      Hub_height -> DTU_hubheight
      P_rated_kW -> DTU_rpower
      Diameter_m -> DTU_diameter
      Commission -> DTU_cdate
    """
    rename_map = {
        "Hub_height": "DTU_hubheight",
        "P_rated_kW": "DTU_rpower",
        "Diameter_m": "DTU_diameter",
        "Commission": "DTU_cdate",
    }
    gdf = gdf.rename(columns=rename_map)

    # Keep only renamed columns + geometry
    keep_cols = list(rename_map.values()) + [gdf.geometry.name]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing_cols]

    return gdf


def process_OSMturbines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only geometry and rename:
      hubheight -> OSM_hubheight
      generator -> OSM_model
      ratedKW   -> OSM_rpower
      rotordia  -> OSM_diameter
    """
    rename_map = {
        "hubheight": "OSM_hubheight",
        "generator": "OSM_model",
        "ratedKW": "OSM_rpower",
        "rotordia": "OSM_diameter",
    }
    gdf = gdf.rename(columns=rename_map)

    # Keep only renamed columns + geometry
    keep_cols = list(rename_map.values()) + [gdf.geometry.name]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing_cols]

    return gdf


def main():
    # Define output directory
    OUTPUT_DIR = "data/00_processedturbines"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Directories
    turbines_dir = "data/TURBINES"
    dtu_dir = "data/DTU_OSM"

    # ----------------------------------------------------------------------
    # 1) Read, reproject, and process DTU/OSM turbines
    # ----------------------------------------------------------------------
    dtuturbines = gpd.read_file(os.path.join(dtu_dir, "DTUturbines.shp"))
    osmturbines = gpd.read_file(os.path.join(dtu_dir, "OSMturbines.shp"))

    dtuturbines.to_crs(epsg=3035, inplace=True)
    osmturbines.to_crs(epsg=3035, inplace=True)

    dtuturbines_processed = process_DTUturbines(dtuturbines)
    osmturbines_processed = process_OSMturbines(osmturbines)

    # ----------------------------------------------------------------------
    # 2) Loop for "new" and "all" datasets
    # ----------------------------------------------------------------------
    for dataset_type in ["new", "all"]:
        pattern = os.path.join(turbines_dir, f"**/*{dataset_type}.shp")
        shp_files = glob(pattern, recursive=True)

        gdf_list = []
        for f in shp_files:
            tmp = gpd.read_file(f)
            if tmp.crs is None:
                tmp.set_crs(epsg=4326, inplace=True)
            tmp.to_crs(epsg=3035, inplace=True)
            gdf_list.append(tmp)

        if gdf_list:
            combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        else:
            # If no files found, create an empty GDF with a known CRS
            combined_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3035")

        # Keep only relevant columns
        geom_col = combined_gdf.geometry.name
        combined_gdf = combined_gdf.loc[
            :, combined_gdf.columns.intersection(["ct_code", "tile_best", "tile", geom_col])
        ]

        # ------------------------------------------------------------------
        # 2a) Map in DTU turbine columns: nearest if within 1500 m
        # ------------------------------------------------------------------
        if not combined_gdf.empty:
            combined_gdf = gpd.sjoin_nearest(
                combined_gdf,
                dtuturbines_processed,
                how="left",
                max_distance=1500,  # 1500 meters
            )
            if "index_right" in combined_gdf.columns:
                combined_gdf.drop(columns=["index_right"], inplace=True)

        # ------------------------------------------------------------------
        # 2b) Map in OSM turbine columns: nearest if within 1500 m
        # ------------------------------------------------------------------
        if not combined_gdf.empty:
            combined_gdf = gpd.sjoin_nearest(combined_gdf, osmturbines_processed, how="left", max_distance=1500)
            if "index_right" in combined_gdf.columns:
                combined_gdf.drop(columns=["index_right"], inplace=True)

        # ------------------------------------------------------------------
        # 2c) Create the 5 new fields:
        #     hubheight, diameter, rpower, model, cdate
        # ------------------------------------------------------------------
        if not combined_gdf.empty:
            # 1) hubheight: copy from OSM, overwrite with DTU if present
            combined_gdf["hubheight"] = combined_gdf["OSM_hubheight"]
            mask_hh = ~combined_gdf["DTU_hubheight"].isna()
            combined_gdf.loc[mask_hh, "hubheight"] = combined_gdf.loc[mask_hh, "DTU_hubheight"]

            # 2) diameter
            combined_gdf["diameter"] = combined_gdf["OSM_diameter"]
            mask_diam = ~combined_gdf["DTU_diameter"].isna()
            combined_gdf.loc[mask_diam, "diameter"] = combined_gdf.loc[mask_diam, "DTU_diameter"]

            # 3) rpower
            combined_gdf["rpower"] = combined_gdf["OSM_rpower"]
            mask_rp = ~combined_gdf["DTU_rpower"].isna()
            combined_gdf.loc[mask_rp, "rpower"] = combined_gdf.loc[mask_rp, "DTU_rpower"]

            # 4) model
            combined_gdf["model"] = combined_gdf["OSM_model"]

            # 5) cdate
            combined_gdf["cdate"] = combined_gdf["DTU_cdate"]

        # ------------------------------------------------------------------
        # 2d) Write out the final "complete_{dataset_type}.gpkg"
        # ------------------------------------------------------------------
        out_file = os.path.join(OUTPUT_DIR, f"complete_{dataset_type}.gpkg")
        combined_gdf.to_file(out_file, driver="GPKG")

    # ----------------------------------------------------------------------
    # 3) Write the final DTU & OSM GPKGs
    # ----------------------------------------------------------------------
    dtuturbines_processed.to_file(os.path.join(OUTPUT_DIR, "DTUturbines.gpkg"), driver="GPKG")
    osmturbines_processed.to_file(os.path.join(OUTPUT_DIR, "OSMturbines.gpkg"), driver="GPKG")


if __name__ == "__main__":
    main()
