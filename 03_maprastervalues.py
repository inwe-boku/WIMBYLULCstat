import os

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from tqdm import tqdm

# countries = ["test"]
# Define input variables

countries = [
    "austria",
    "belgium",
    "denmark",
    "estonia",
    "france",
    "germany",
    "greece",
    "ireland",
    "italy",
    "latvia",
    "lithuania",
    "luxemburg",
    "netherlands",
    "norway",
    "poland",
    "portugal",
    "spain",
    "sweden",
    "unitedkingdom",
]  # Add other countries as needed
datasets = ["new", "all", "old"]
input_base_dir = "data/02_outhulls"
output_base_dir = "data/03_outraster"
raster_files = {
    "altitude": {"file": "Copernicus_DSM_90m_COG_3035.tif", "precision": 0},
    "slope": {"file": "Copernicus_SLOPE_90m_COG_3035.tif", "precision": 1},
    "tri": {"file": "Copernicus_TRI_90m_COG_3035.tif", "precision": 2},
}
polytypes = ["turbines", "bufferpolys", "concavehulls", "convexhulls"]

# Debug flag
debug = False  # Set to False to suppress debug outputs


# Function to calculate mean raster value for a polygon
def calculate_mean_raster_value(raster, geom, pixel_size=None):
    try:
        masked, _ = rasterio.mask.mask(raster, [geom], crop=True, filled=True, nodata=np.nan)
        data = masked[0]  # First band
        if np.isnan(data).all():  # Check if the array is entirely NaN
            # Apply buffering if no data found
            if pixel_size is not None:
                buffered_geom = geom.buffer(pixel_size / 2)  # Buffer half pixel size
                masked, _ = rasterio.mask.mask(raster, [buffered_geom], crop=True, filled=True, nodata=np.nan)
                data = masked[0]  # Check buffered data
            if np.isnan(data).all():  # Still all NaN, return NaN
                return np.nan
        with np.errstate(invalid="ignore"):  # Suppress warnings for mean of empty slices
            mean_value = np.nanmean(data)  # Compute mean excluding nodata values
        return mean_value
    except Exception as e:
        if debug:
            print(f"Error sampling raster for geometry: {e}")
        return np.nan  # Return NaN for any errors


# Outer progress bar for countries
with tqdm(total=len(countries), desc="Processing countries") as country_bar:
    for country in countries:
        input_dir = os.path.join(input_base_dir, country)
        output_dir = os.path.join(output_base_dir, country)
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

        # Total progress for datasets, rasters, and layers
        total_tasks = len(datasets) * (len(raster_files) * len(polytypes))
        with tqdm(total=total_tasks, desc=f"Processing {country}", leave=False) as task_bar:
            for dataset in datasets:
                gpkg_path = os.path.join(input_dir, f"{country}_{dataset}.gpkg")
                if not os.path.exists(gpkg_path):
                    if debug:
                        print(f"File not found: {gpkg_path}")
                    continue

                # Read all layers and compute a bounding box
                bounding_box = None
                gdfs = {}
                for polytype in polytypes:
                    try:
                        gdf = gpd.read_file(gpkg_path, layer=polytype)
                        gdfs[polytype] = gdf
                        if bounding_box is None:
                            bounding_box = gdf.total_bounds
                        else:
                            bounding_box = [
                                min(bounding_box[0], gdf.total_bounds[0]),
                                min(bounding_box[1], gdf.total_bounds[1]),
                                max(bounding_box[2], gdf.total_bounds[2]),
                                max(bounding_box[3], gdf.total_bounds[3]),
                            ]
                    except Exception as e:
                        if debug:
                            print(f"Error reading layer {polytype} in {gpkg_path}: {e}")
                    task_bar.update(1)

                if bounding_box is None:
                    if debug:
                        print(f"No valid data in {gpkg_path}")
                    continue

                # Loop through raster files
                for field_name, raster_info in raster_files.items():
                    raster_path = os.path.join("data/RASTER", raster_info["file"])
                    precision = raster_info["precision"]

                    with rasterio.open(raster_path) as raster:
                        pixel_size = raster.res[0]  # Get pixel size (assumes square pixels)

                        # Ensure CRS matches
                        for polytype, gdf in gdfs.items():
                            if gdf.crs != raster.crs:
                                gdf = gdf.to_crs(raster.crs)

                        window = raster.window(*bounding_box)
                        transform = raster.window_transform(window)
                        raster_data = raster.read(1, window=window, masked=True)  # Read the first band

                        for polytype, gdf in gdfs.items():
                            # Determine geometry type dynamically
                            geom_type = gdf.geometry.geom_type.unique()
                            if len(geom_type) > 1:
                                raise ValueError(
                                    f"Mixed geometry types in layer '{polytype}'. Found types: {geom_type}"
                                )
                            geom_type = geom_type[0].lower()

                            if geom_type == "point":  # Point data
                                values = []
                                for point in gdf.geometry:
                                    try:
                                        # Convert point coordinates to raster indices
                                        row, col = raster.index(point.x, point.y)
                                        # Adjust for the window offset
                                        row_offset = int(row - window.row_off)
                                        col_offset = int(col - window.col_off)
                                        # Sample the raster value
                                        if (
                                            0 <= row_offset < raster_data.shape[0]
                                            and 0 <= col_offset < raster_data.shape[1]
                                        ):
                                            value = float(raster_data[row_offset, col_offset])  # Ensure float
                                            if precision == 0:
                                                values.append(int(value))
                                            else:
                                                values.append(round(value, precision))
                                        else:
                                            values.append(np.nan)
                                    except Exception as e:
                                        if debug:
                                            print(f"Error sampling point at {point}: {e}")
                                        values.append(np.nan)
                                gdf[field_name] = values
                            elif geom_type in {
                                "polygon",
                                "multipolygon",
                            }:  # Polygon data

                                def process_polygon(geom):
                                    value = calculate_mean_raster_value(raster, geom, pixel_size=pixel_size)
                                    if not np.isnan(value):
                                        value = float(value)  # Ensure float
                                        if precision == 0:
                                            return int(value)
                                        return round(value, precision)
                                    return np.nan

                                gdf[field_name] = gdf.geometry.apply(process_polygon)
                            else:
                                if debug:
                                    print(f"Unsupported geometry type '{geom_type}' in layer '{polytype}'")
                            task_bar.update(1)

                # Save updated GeoDataFrames back to the GeoPackage
                for polytype, gdf in gdfs.items():
                    output_layer_path = os.path.join(output_dir, f"{country}_{dataset}.gpkg")
                    gdf.to_file(output_layer_path, layer=polytype, driver="GPKG")

                    # Compare feature counts if debug is active
                    if debug:
                        input_feature_count = len(gdf)
                        output_feature_count = len(gdf)
                        print(f"Feature count for layer '{polytype}' in {gpkg_path}:")
                        print(f"  Input features: {input_feature_count}")
                        print(f"  Output features: {output_feature_count}")

                        if input_feature_count != output_feature_count:
                            print(f"WARNING: Feature count mismatch in layer '{polytype}'!")
                        else:
                            print(f"Feature count matches for layer '{polytype}'.")
                    task_bar.update(1)
        country_bar.update(1)
