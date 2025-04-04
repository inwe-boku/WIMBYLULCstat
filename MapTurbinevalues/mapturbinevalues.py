import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# pd.set_option('display.max_rows', None)

maxdistance = 200


def read_all_shp(directory, pattern):
    """Read all shapefiles matching the pattern from a directory recursively and concatenate into one GeoDataFrame."""
    files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    gdfs = []
    for file in files:
        gdf = gpd.read_file(file)
        gdf = gdf.to_crs("epsg:3035")
        gdfs.append(gdf)
    return pd.concat(gdfs, ignore_index=True)


# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
turbine_dir = os.path.join(script_dir, "../data/TURBINES")

# Read the WIMBY turbine files
wimby_new_turbines = read_all_shp(turbine_dir, "*new.shp")
wimby_all_turbines = read_all_shp(turbine_dir, "*all.shp")

# Read the shapefiles
dtuturbines = gpd.read_file(script_dir + "/DTUturbines.shp")
osmturbines = gpd.read_file(script_dir + "/OSMturbines.shp")

# Extract coordinates
dtuturbines = dtuturbines.to_crs("epsg:3857")
dtucoords = dtuturbines.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
dtutree = cKDTree(dtucoords)

osmturbines = osmturbines.to_crs("epsg:3857")
osmcoords = osmturbines.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
osmtree = cKDTree(osmcoords)


# Process turbines with DTU and OSM information
def process_turbines(wimbyturbines, filename):
    wimbyturbines = wimbyturbines.to_crs("epsg:3857")
    wimbycoords = wimbyturbines.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    # Find nearest neighbors
    dtudistances, dtuindices = dtutree.query(wimbycoords, distance_upper_bound=maxdistance)
    dtumask = np.isfinite(dtudistances)
    wimbyindices = np.arange(len(wimbyturbines))[dtumask]
    dtuindices = dtuindices[dtumask]

    for wimbyidx, dtuidx in zip(wimbyindices, dtuindices):
        for column in dtuturbines.columns:
            if column != "geometry":
                wimbyturbines.at[wimbyidx, "DTU_hubh"] = dtuturbines.at[dtuidx, "Hub_height"]
                wimbyturbines.at[wimbyidx, "DTU_power"] = dtuturbines.at[dtuidx, "P_rated_kW"]
                wimbyturbines.at[wimbyidx, "DTU_dia"] = dtuturbines.at[dtuidx, "Diameter_m"]
                wimbyturbines.at[wimbyidx, "DTU_date"] = dtuturbines.at[dtuidx, "Commission"]

    osmdistances, osmindices = osmtree.query(wimbycoords, distance_upper_bound=maxdistance)
    osmmask = np.isfinite(osmdistances)
    wimbyindices = np.arange(len(wimbyturbines))[osmmask]
    osmindices = osmindices[osmmask]

    for wimbyidx, osmidx in zip(wimbyindices, osmindices):
        for column in osmturbines.columns:
            if column != "geometry":
                wimbyturbines.at[wimbyidx, "OSM_hubh"] = osmturbines.at[osmidx, "hubheight"]
                wimbyturbines.at[wimbyidx, "OSM_power"] = osmturbines.at[osmidx, "ratedKW"]
                wimbyturbines.at[wimbyidx, "OSM_dia"] = osmturbines.at[osmidx, "rotordia"]
                wimbyturbines.at[wimbyidx, "OSM_manu"] = osmturbines.at[osmidx, "generator"]

    wimbyturbines["hubheight"] = np.where(
        wimbyturbines["DTU_hubh"].notna(),
        wimbyturbines["DTU_hubh"],
        wimbyturbines["OSM_hubh"],
    )
    wimbyturbines["power"] = np.where(
        wimbyturbines["DTU_power"].notna(),
        wimbyturbines["DTU_power"],
        wimbyturbines["OSM_power"],
    )
    wimbyturbines["diameter"] = np.where(
        wimbyturbines["DTU_dia"].notna(), wimbyturbines["DTU_dia"], wimbyturbines["OSM_dia"]
    )
    wimbyturbines["date"] = wimbyturbines["DTU_date"]
    wimbyturbines["manufactur"] = wimbyturbines["OSM_manu"]

    # Save the result
    # Drop problematic columns like 'fid'
    wimbyturbines = wimbyturbines.loc[:, ~wimbyturbines.columns.str.contains("^fid$", case=False)]
    # Save the result
    wimbyturbines.to_file(script_dir + f"/{filename}.gpkg", driver="GPKG")


# Process both datasets
process_turbines(wimby_new_turbines, "WIMBYturbines_new_DTU_OSM")
process_turbines(wimby_all_turbines, "WIMBYturbines_all_DTU_OSM")
wimbyturbines.to_file(script_dir + "/WIMBYturbines_new_DTU_OSM.shp")
