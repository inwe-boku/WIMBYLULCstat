import argparse
import os
import re

import geopandas as gpd
import numpy as np

# from shapely.wkt import loads
import pandas as pd
import rasterio

# from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import Point, box

debug = True
project_crs = "EPSG:3035"
clc_files = {
    "clc100": {"file": "U2018_CLC2018_V2020_20u1.tif"},
    "clc10": {"file": "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif"},
}

# countries = ["france"] #["austria", "france"]
datasets = ["old"]


# Function to read raster files and extract their bounds
def read_raster_bounds(cdtile_path):
    tile_data = []
    tile_crs = None

    for raster_file in sorted(os.listdir(cdtile_path)):
        if raster_file.endswith("_binary.tif"):
            raster_path = os.path.join(cdtile_path, raster_file)
            with rasterio.open(raster_path) as src:
                bounds = box(*src.bounds)
                tile_name = re.match(r"^([0-9A-Z]+)", raster_file).group(1)
                tile_crs = src.crs.to_string()
                tile_data.append({"tile_name": tile_name, "bounds": bounds, "crs": tile_crs})

    tile_df = pd.DataFrame(tile_data)
    return tile_df


def process_hulls_and_tiles(hulls: gpd.GeoDataFrame, tile_df: pd.DataFrame):
    """
    Process and clip hulls to tiles, matching and calculating areas.

    Parameters:
        hulls (gpd.GeoDataFrame): GeoDataFrame with polygons and 'tile_best' column.
        tile_df (pd.DataFrame): DataFrame with columns 'tile_name', 'bounds' (shapely Polygons), and 'crs'.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of processed hulls with added columns.
    """
    # Add area and hull_id to the hulls GeoDataFrame
    hulls = hulls.copy()
    hulls["area"] = hulls.geometry.area.astype(int)
    hulls["hull_id"] = hulls.index

    hulls_best = []

    for _, tile in tile_df.iterrows():
        tile_name = tile["tile_name"]
        tile_bounds = tile["bounds"]
        tile_crs = tile["crs"]

        # Ensure hulls are in the tile CRS
        hulls_in_tile_crs = hulls.to_crs(tile_crs)

        # Clip hulls to the tile bounds
        tile_polygon = gpd.GeoDataFrame({"geometry": [tile_bounds]}, crs=tile_crs)
        clipped_hulls = gpd.overlay(hulls_in_tile_crs, tile_polygon, how="intersection")

        hulls_tile = []

        # Loop through the clipped hulls
        for idx, clipped_hull in clipped_hulls.iterrows():
            original_hull = hulls.loc[clipped_hull["hull_id"]]

            if original_hull["tile_best"] == tile_name:
                # Calculate area of the clipped feature
                areac = clipped_hull.geometry.area
                areap = areac / original_hull["area"] if original_hull["area"] > 0 else 0

                # Add the processed hull to the list
                clipped_hull["areac"] = int(areac)
                clipped_hull["areap"] = round(areap * 100, 2)
                clipped_hull["tile_sample"] = tile_name
                hulls_tile.append(clipped_hull)

        # Combine all processed hulls into a new GeoDataFrame
        hulls_tile = pd.DataFrame(hulls_tile)
        if hulls_tile.empty:
            continue
        hulls_tile_gdf = gpd.GeoDataFrame(hulls_tile, geometry="geometry", crs=tile_crs)
        hulls_tile_gdf = hulls_tile_gdf.to_crs(project_crs)
        hulls_best.append(hulls_tile_gdf)

    if hulls_best:
        hulls_best = pd.concat(hulls_best, ignore_index=True)
        hulls_best = gpd.GeoDataFrame(hulls_best, geometry="geometry", crs=project_crs)

        # Save to file only if hulls_best is not empty
        if not hulls_best.empty:
            hulls_best.to_file("data/debug/hulls_best.gpkg")

            # Reproject hulls_best to match the CRS of hulls
            hulls_best = hulls_best.to_crs(hulls.crs)

            # Get unique hull IDs
            hulls_best_ids = hulls_best["hull_id"].unique()

            # Perform the geometric difference
            hulls["geometry"] = hulls.geometry.apply(lambda geom: geom.difference(hulls_best.buffer(0.5).union_all()))
    else:
        # If hulls_best is empty, just pass the original geometries
        hulls_best_ids = []
        print("hulls_best is empty, no difference calculation performed.")
    # Remove any empty geometries resulting from the operation
    hulls = hulls[hulls.geometry.is_empty == False]

    hulls.to_file("data/debug/hulls_remains.gpkg")

    hulls_remainc = []

    for _, tile in tile_df.iterrows():
        tile_name = tile["tile_name"]
        tile_bounds = tile["bounds"]
        tile_crs = tile["crs"]

        # Ensure hulls are in the tile CRS
        hulls_in_tile_crs = hulls.to_crs(tile_crs)

        # Clip hulls to the tile bounds
        tile_polygon = gpd.GeoDataFrame({"geometry": [tile_bounds]}, crs=tile_crs)
        clipped_hulls = gpd.overlay(hulls_in_tile_crs, tile_polygon, how="intersection")

        hulls_tile = []

        # Loop through the clipped hulls
        for idx, clipped_hull in clipped_hulls.iterrows():
            original_hull = hulls.loc[clipped_hull["hull_id"]]

            areac = clipped_hull.geometry.area
            areap = areac / original_hull["area"] if original_hull["area"] > 0 else 0

            clipped_hull["areac"] = int(areac)
            clipped_hull["areap"] = round(areap * 100, 2)

            if clipped_hull["hull_id"] in hulls_best_ids:
                # Add the processed hull to the list
                clipped_hull["tile_sample"] = tile_name
                hulls_tile.append(clipped_hull)

            if clipped_hull["areap"] > 99.9:
                # Add the processed hull to the list
                clipped_hull["tile_sample"] = tile_name
                hulls_tile.append(clipped_hull)

        # Combine all processed hulls into a new GeoDataFrame
        hulls_tile = pd.DataFrame(hulls_tile)
        if hulls_tile.empty:
            continue

        hulls_tile_gdf = gpd.GeoDataFrame(hulls_tile, geometry="geometry", crs=tile_crs)
        hulls_tile_gdf = hulls_tile_gdf.to_crs(project_crs)

        hulls_remainc.append(hulls_tile_gdf)
    if len(hulls_remainc) > 0:
        hulls_remainc = pd.concat(hulls_remainc, ignore_index=True)
        hulls_remainc = gpd.GeoDataFrame(hulls_remainc, geometry="geometry", crs=project_crs)

        hulls_remainc.to_file("data/debug/hulls_remain_clipped.gpkg")
        return gpd.GeoDataFrame(pd.concat([hulls_best, hulls_remainc], ignore_index=True))
    else:
        return hulls_best


def mask_raster_with_polygon_and_recode(tile_name, raster_path, polygon_gdf):
    """
    Masks the raster using polygons and recodes pixel values, but only for rows in
    polygon_gdf where the 'tile_sample' column matches the given tile_name.
    """
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        raster_meta = src.meta.copy()

        # Filter polygons for the matching tile_name
        matching_polygons = polygon_gdf[polygon_gdf["tile_sample"] == tile_name]

        # If no matching polygons, return an empty masked raster
        if matching_polygons.empty:
            return np.zeros_like(raster), src.transform, src.crs

        # Reproject polygons to match the raster CRS
        matching_polygons = matching_polygons.to_crs(src.crs)

        # Extract geometries for masking
        polygons = [geometry for geometry in matching_polygons.geometry]

        # Create mask
        mask = geometry_mask(
            polygons,
            transform=src.transform,
            invert=True,
            out_shape=raster.shape,
            all_touched=True,
        )

        # Initialize masked raster
        masked_raster = np.zeros_like(raster)

        # Perform recoding only for matching tiles
        masked_raster[mask & (raster == 1)] = 1
        masked_raster[mask & (raster == 0)] = 2

    return masked_raster, src.transform, src.crs


def clean_narrow_parts(gdf, min_width=10):
    """
    Remove narrow parts of polygons in a GeoDataFrame.

    Parameters:
    - gdf: GeoDataFrame with polygon geometries.
    - min_width: Minimum width to retain, in the same units as the GeoDataFrame's CRS.

    Returns:
    - GeoDataFrame with cleaned geometries.
    """
    half_width = min_width / 2
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: geom.buffer(half_width).buffer(-half_width) if geom.is_valid else geom.buffer(0)
    )
    return gdf


def raster_to_point_gdf(masked_raster, transform, crs):
    """Converts raster data to a GeoDataFrame of points."""
    rows, cols = np.where((masked_raster == 1) | (masked_raster == 2))
    pixel_values = masked_raster[rows, cols]
    coords = [rasterio.transform.xy(transform, row, col) for row, col in zip(rows, cols)]
    points = [Point(x, y) for x, y in coords]
    data = {"geometry": points, "CD": pixel_values}
    return gpd.GeoDataFrame(data, crs=crs)


def sample_raster_with_points(raster_path, points_gdf, clc_key):
    """Samples raster values at the locations specified in the point GeoDataFrame."""
    with rasterio.open(raster_path) as src:
        points_gdf = points_gdf.to_crs(src.crs)
        raster_values = [val[0] for point in points_gdf.geometry for val in src.sample([(point.x, point.y)])]
    points_gdf[clc_key] = raster_values
    points_gdf = points_gdf.to_crs(project_crs)
    return points_gdf


def compute_statistics_by_hull_id(points_gdf, clc_key):
    """
    Computes statistics by `hull_id` where:
    - If CD = 1, count up a new field called CD_$rastervalue.
    - If CD = 2, count up a new field called VAL_$rastervalue.
    Aggregates these counts by `hull_id`.

    Parameters:
        points_gdf (GeoDataFrame): A GeoDataFrame with fields "CD", "raster_value", and "hull_id".

    Returns:
        DataFrame: A DataFrame with aggregated statistics by `hull_id`.
    """
    # Ensure input fields exist
    required_fields = ["CD", clc_key, "hull_id"]
    for field in required_fields:
        if field not in points_gdf.columns:
            raise ValueError(f"Missing required field '{field}' in points GeoDataFrame.")

    # Define a function to calculate the statistics for each hull_id group
    def calculate_group_stats(group, clc_key):
        stats = {}
        val_counts = {}  # Temporary dictionary to accumulate VAL counts

        for _, row in group.iterrows():
            if row["CD"] == 1:
                # Count for CD
                key_cd = f"{clc_key}_CD_{row[clc_key]}"
                stats[key_cd] = stats.get(key_cd, 0) + 1

            # Accumulate VAL counts for all raster_value
            key_val = f"{clc_key}_VAL_{row[clc_key]}"
            val_counts[key_val] = val_counts.get(key_val, 0) + 1

        # Add accumulated VAL counts to stats
        stats.update(val_counts)

        return stats

    # Group by `hull_id` and apply the stats function
    grouped = points_gdf.groupby("hull_id")
    stats_list = []
    for hull_id, group in grouped:
        stats = calculate_group_stats(group, clc_key)
        stats["hull_id"] = hull_id
        stats_list.append(stats)

    # Convert list of dictionaries into a DataFrame
    stats_df = pd.DataFrame(stats_list).fillna(0)
    return stats_df


def drop_duplicate_columns_from_dict(dataframes_dict):
    seen_columns = set()
    updated_dataframes_dict = {}

    for key, df in dataframes_dict.items():
        if not updated_dataframes_dict:  # For the first DataFrame
            updated_dataframes_dict[key] = df
            seen_columns.update(df.columns)
        else:
            # Retain only columns not seen before
            unique_columns = [col for col in df.columns if col not in seen_columns]
            updated_dataframes_dict[key] = df[unique_columns]
            seen_columns.update(df.columns)

    return updated_dataframes_dict


def combine_clc_statistics(all_clc_results):
    """Combine CLC statistics into a single GeoDataFrame with both clc100 and clc10 columns."""
    merged_result = None
    all_clc_results = drop_duplicate_columns_from_dict(all_clc_results)
    for clc_key, result_gdf in all_clc_results.items():
        if merged_result is None:
            merged_result = result_gdf
        else:
            # Merge on the polygon geometry or a unique ID (if available)
            merged_result = merged_result.merge(
                result_gdf,  # Drop geometry to avoid duplicate columns
                left_index=True,
                right_index=True,
                how="outer",
                suffixes=("", f"_{clc_key}"),
            )
    return merged_result


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process CLC data for a list of countries.")
    parser.add_argument(
        "--countries",
        type=str,
        required=True,
        help="Comma-separated list of countries to process (e.g., 'austria,france').",
    )
    return parser.parse_args()


def clean_and_sort(hulls):
    # Rename columns only if they exist
    if "cluster" not in hulls.columns:
        if "cls_new" in hulls.columns:
            hulls.rename(columns={"cls_new": "cluster"}, inplace=True)

    if "buffer" not in hulls.columns:
        if "buf_new" in hulls.columns:
            hulls.rename(columns={"buf_new": "buffer"}, inplace=True)

    # Drop columns only if they exist
    columns_to_drop = ["split_area", "original_area", "area_percentage"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in hulls.columns]
    hulls.drop(columns=existing_columns_to_drop, inplace=True)

    # Separate columns into `clc` columns and others
    clc_columns = [col for col in hulls.columns if col.startswith("clc")]
    other_columns = [col for col in hulls.columns if not col.startswith("clc")]

    # Sort `clc` columns with the correct order
    sorted_clc_columns = sorted(
        clc_columns,
        key=lambda x: (
            x.split("_")[0],  # First segment (e.g., clc100, clc10)
            int(x.split("_")[-1]),  # Last segment (numerical value)
            x.split("_")[1],  # Middle segment (e.g., VAL before CD)
        ),
    )

    # Combine sorted columns
    sorted_columns = other_columns + sorted_clc_columns

    # Reorder DataFrame
    hulls = hulls[sorted_columns]
    return hulls


# Main script
def main():
    args = parse_arguments()
    countries = args.countries.split(",")
    for country in countries:
        for dataset in datasets:
            capitalized_country = country.capitalize()
            cdtile_path = f"data/CDTiles/{capitalized_country}/"
            gpkg_path = f"data/03_outraster/{country.lower()}/{country.lower()}_{dataset}.gpkg"
            output_gdf_path = f"data/04_clc/{country.lower()}/{country.lower()}_{dataset}.gpkg"

            # Ensure output directory exists
            output_dir = os.path.dirname(output_gdf_path)
            os.makedirs(output_dir, exist_ok=True)

            samplelayer = "convexhulls"
            layers = ["bufferpolys", "concavehulls", "convexhulls"]

            if debug:
                print(f"Processing country: {country}, dataset: {dataset}")
                print(f"Reading raster bounds from: {cdtile_path}")

            # Read raster bounds
            tile_df = read_raster_bounds(cdtile_path)

            if debug:
                print(f"Found {len(tile_df)} tiles in {cdtile_path}")

            # Process each layer
            # Load hulls for the current layer
            hulls = gpd.read_file(gpkg_path, layer=samplelayer)
            # hulls_file = 'data/debug/hulls.gpkg'
            # hulls.to_file(hulls_file, layer=samplelayer, driver="GPKG")

            if debug:
                print(f"Loaded {len(hulls)} polygons from layer: {samplelayer}")

            # Process polygons and calculate differences
            combined_hulls = process_hulls_and_tiles(hulls, tile_df)
            # Iterate over CDTiles based on original filenames
            all_points = []

            for _, tile in tile_df.iterrows():
                tile_name = tile["tile_name"]
                raster_filename = next(
                    (f for f in os.listdir(cdtile_path) if f.startswith(tile_name) & f.endswith("binary.tif")),
                    None,
                )

                if not raster_filename:
                    if debug:
                        print(f"No matching raster file for tile: {tile_name}, skipping...")
                    continue

                cdtilefilepath = os.path.join(cdtile_path, raster_filename)

                if debug:
                    print(f"Processing tile: {tile_name} with raster: {raster_filename}")

                # Mask raster with polygons and recode
                masked_raster, transform, crs = mask_raster_with_polygon_and_recode(
                    tile_name, cdtilefilepath, combined_hulls
                )

                # Convert masked raster to point GeoDataFrame
                points_gdf = raster_to_point_gdf(masked_raster, transform, crs)

                if points_gdf.empty:
                    if debug:
                        print(f"No points generated for tile: {tile_name}")
                    continue
                points_gdf = points_gdf.to_crs(project_crs)

                all_points.append(points_gdf)

            if debug:
                print(f"Finished processing all tiles for {country} - {dataset}")

            all_points = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True))

            # all_points = all_points.iloc[:100000]

            for clc_key, clc_data in clc_files.items():
                lulcfile = f"data/CLC/{clc_data['file']}"
                if debug:
                    print(f"Sampling raster values from {lulcfile} for {clc_key}")
                all_points = sample_raster_with_points(lulcfile, all_points, clc_key)

            # all_points.to_file(output_gdf_path + "_points.gpkg", driver="GPKG")
            turbines = gpd.read_file(gpkg_path, layer="turbines")
            turbines.to_file(output_gdf_path, layer="turbines", driver="GPKG")

            for layer in layers:
                if debug:
                    print(f"Processing layer: {layer}")
                hulls = gpd.read_file(gpkg_path, layer=layer)
                hulls["hull_id"] = hulls.index
                hulls = hulls.to_crs(project_crs)
                all_points = all_points.to_crs(project_crs)

                joined_gdf = gpd.sjoin(
                    all_points,
                    hulls[["hull_id", "geometry"]],
                    how="left",
                    predicate="within",
                )
                all_stats = pd.DataFrame()
                for clc_key, clc_data in clc_files.items():
                    stats_df = compute_statistics_by_hull_id(joined_gdf, clc_key)
                    if all_stats.empty:
                        all_stats = stats_df
                    else:
                        all_stats = pd.merge(all_stats, stats_df, on="hull_id")

                clc_results = hulls.merge(all_stats, on="hull_id", how="left").fillna(0)

                if debug:
                    print(f"Saving processed results for layer: {layer} to {output_gdf_path}")
                clc_results["area"] = clc_results.geometry.area.astype(int)
                clc_results = clean_and_sort(clc_results)
                # print(clc_results.columns)
                clc_results.to_file(output_gdf_path, layer=layer, driver="GPKG")


if __name__ == "__main__":
    main()
