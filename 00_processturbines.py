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
        "Commission": "DTU_cdate"
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
        "rotordia": "OSM_diameter"
    }
    gdf = gdf.rename(columns=rename_map)

    # Keep only renamed columns + geometry
    keep_cols = list(rename_map.values()) + [gdf.geometry.name]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing_cols]

    return gdf

def main():
    # Directories
    turbines_dir = "data/TURBINES"
    dtu_dir = "data/DTU_OSM"

    # 1) Read DTU and OSM turbines
    dtuturbines = gpd.read_file(os.path.join(dtu_dir, "DTUturbines.shp"))
    osmturbines = gpd.read_file(os.path.join(dtu_dir, "OSMturbines.shp"))

    # 2) Reproject DTU and OSM turbines in place to EPSG:3035
    dtuturbines.to_crs(epsg=3035, inplace=True)
    osmturbines.to_crs(epsg=3035, inplace=True)

    # 3) Process DTU and OSM turbines
    dtuturbines_processed = process_DTUturbines(dtuturbines)
    osmturbines_processed = process_OSMturbines(osmturbines)

    # 4) Loop for "new" and "all" datasets
    for dataset_type in ["new", "all"]:
        pattern = os.path.join(turbines_dir, f"**/*{dataset_type}.shp")
        gpkg_files = glob(pattern, recursive=True)

        gdf_list = []
        for f in gpkg_files:
            tmp = gpd.read_file(f)
            # If CRS is missing, set it to whatever the correct source CRS is
            if tmp.crs is None:
                # Replace EPSG:4326 with the actual source CRS
                tmp.set_crs(epsg=4326, inplace=True)
            # Now reproject
            tmp.to_crs(epsg=3035, inplace=True)

            gdf_list.append(tmp)

        if gdf_list:
            combined_gdf = pd.concat(gdf_list, ignore_index=True)
            combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        else:
            # If no files found, create an empty GDF with a known CRS
            combined_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3035")

        # Write out
        geom_col = combined_gdf.geometry.name
        # Then just pick the columns we want:
        combined_gdf = combined_gdf.loc[:, combined_gdf.columns.intersection(['ct_code','tile_best', "tile", geom_col])]
        combined_gdf.to_file(f"complete_{dataset_type}.gpkg", driver="GPKG")

    # 5) After processing "new" and "all", write the final DTU & OSM GPKGs
    dtuturbines_processed.to_file(
        os.path.join(dtu_dir, "DTUturbines.gpkg"), driver="GPKG"
    )
    osmturbines_processed.to_file(
        os.path.join(dtu_dir, "OSMturbines.gpkg"), driver="GPKG"
    )

if __name__ == "__main__":
    main()
