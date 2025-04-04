import os

import geopandas as gpd
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.ops import split
from sklearn.cluster import DBSCAN
from tqdm import tqdm  # For progress bars

# Constants
CRS = "epsg:3035"
EPSILON = 1500  # DBSCAN distance threshold
MIN_SAMPLES = 2  # Minimum samples for DBSCAN clustering
BUFFERPERC = 0.5  # Buffer percentage
BUFFERSCALE1 = 2 / 3
BUFFERSCALE2 = 1 / 2
SAMPLE_POINTS = 99
CONCAVE_RATIO = 0.25


def clustercalculation(turbines):
    coords = turbines.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    dbscan = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES)
    clusters = dbscan.fit_predict(coords)
    turbines["cluster"] = clusters

    # Calculate buffer sizes
    cluster_distances = {}
    for clusternum in np.unique(clusters):
        if clusternum == -1:
            continue  # Skip noise points
        cpoints = turbines[turbines["cluster"] == clusternum]["geometry"]
        ccoords = np.array([(point.x, point.y) for point in cpoints])
        distance_matrix = distance.cdist(ccoords, ccoords)
        distance_matrix[distance_matrix < 9] = np.inf
        min_distances = np.min(distance_matrix, axis=1)

        # Calculate buffer size if there are valid distances
        valid_distances = min_distances[np.isfinite(min_distances)]
        if len(valid_distances) > 0:
            buffer = int(round(np.mean(valid_distances * BUFFERPERC), 0))
        else:
            buffer = np.nan  # Assign NaN if no valid distances exist
        cluster_distances[clusternum] = buffer

    # Assign buffers to points
    turbines["buffer"] = turbines["cluster"].map(cluster_distances)

    # Calculate average buffer size (ignoring NaN) for current dataset
    valid_buffers = turbines["buffer"].dropna()
    if not valid_buffers.empty:
        buffer_default = int(round(valid_buffers.mean(), 0))
    else:
        buffer_default = 0  # Fallback if no valid buffers exist at all (edge case)

    # Replace NaN buffers with the average and reassign to the column
    turbines["buffer"] = turbines["buffer"].fillna(buffer_default)
    return turbines


def create_hulls(points, hull_type="convex"):
    if hull_type == "buffer":
        hull_gdf = create_buffer_polygons(points)
    elif hull_type == "convex":
        hull_gdf = create_convex_hulls(points)
    elif hull_type == "concave":
        hull_gdf = create_concave_hulls2(points)
    else:
        raise ValueError("Invalid hull_type. Choose from 'buffer', 'convex', or 'concave'.")
    hull_gdf.reset_index(inplace=True)
    hull_gdf["hull_id"] = hull_gdf.index + 1  # Start IDs from 1
    return hull_gdf


# Supporting Functions
def create_buffer_polygons(points):
    # Use vectorized buffer operation to create buffer polygons
    bufferpolys = points.copy(deep=False)
    bufferpolys.drop(columns=["geometry"], inplace=True)
    bufferpolys["geometry"] = points.buffer(points["buffer"].values * (BUFFERSCALE1 + BUFFERSCALE2))
    return bufferpolys


def create_convex_hulls(points):
    grouped = points.dissolve(by="cluster")
    buffer_polygons = grouped.convex_hull
    buffer_polygons = buffer_polygons.buffer(grouped["buffer"].values * (BUFFERSCALE1 + BUFFERSCALE2))
    bufferpolys = grouped.copy(deep=False)
    bufferpolys.drop(columns=["geometry"], inplace=True)
    bufferpolys["geometry"] = buffer_polygons
    return bufferpolys


def create_concave_hulls(points):
    grouped = points.dissolve(by="cluster")
    # Assuming the concave hull method is defined elsewhere in the workflow.
    buffer_polygons = grouped.concave_hull()
    buffer_polygons = buffer_polygons.buffer(grouped["buffer"].values * (BUFFERSCALE1 + BUFFERSCALE2))
    bufferpolys = grouped.copy(deep=False)
    bufferpolys.drop(columns=["geometry"], inplace=True)
    bufferpolys["geometry"] = buffer_polygons
    return bufferpolys


def sample_buffer_boundaries(points, buffer_scale=2 / 3, n_points=99):
    sampled_records = []

    for idx, row in points.iterrows():
        buffer_geom = row.geometry.buffer(row["buffer"] * buffer_scale)
        boundary = buffer_geom.boundary

        # Get coordinates from boundary and sample evenly spaced points
        coords = list(boundary.coords)
        step = max(1, len(coords) // n_points)
        sampled_coords = coords[::step]

        for pt in sampled_coords:
            sampled_record = row.to_dict()
            sampled_record["geometry"] = Point(pt)
            sampled_records.append(sampled_record)

    # Create GeoDataFrame from sampled points with all original attributes
    bufferpoints = gpd.GeoDataFrame(sampled_records, crs=points.crs)
    return bufferpoints


def create_concave_hulls2(
    points,
    buffer_scale1=BUFFERSCALE1,
    buffer_scale2=BUFFERSCALE2,
    n_points=SAMPLE_POINTS,
    concave_ratio=CONCAVE_RATIO,
):
    bufferpoints = sample_buffer_boundaries(points, buffer_scale=buffer_scale1, n_points=n_points)
    grouped = bufferpoints.dissolve(by="cluster")
    buffer_polygons = grouped.concave_hull(ratio=concave_ratio)
    buffer_polygons = buffer_polygons.buffer(grouped["buffer"].values * buffer_scale2)
    bufferpolys = grouped.copy(deep=False)
    bufferpolys.drop(columns=["geometry"], inplace=True)
    bufferpolys["geometry"] = buffer_polygons
    return bufferpolys


def split_hulls(hulls, split_polygons):
    """
    Splits hulls using boundaries of polygons and retains attributes from the original dataset,
    ensuring no overlapping or duplicate polygons are retained.

    Parameters:
        hulls (GeoDataFrame): The GeoDataFrame containing polygons to be split.
        split_polygons (GeoDataFrame): The GeoDataFrame containing polygons whose boundaries will be used for splitting.

    Returns:
        GeoDataFrame: Split polygons with attributes retained from the original dataset.
    """
    split_lines = split_polygons.boundary  # Extract boundaries (LineString/MultiLineString)
    split_results = []

    for idx, hull in hulls.iterrows():
        # Start with the initial geometry
        geometries_to_split = [hull.geometry]
        final_geometries = []

        for line in split_lines:
            next_geometries = []
            for geometry in geometries_to_split:
                if geometry.intersects(line):
                    # Split the geometry by the line
                    split_geom = split(geometry, line)
                    next_geometries.extend(geom for geom in split_geom.geoms if geom.is_valid and not geom.is_empty)
                else:
                    next_geometries.append(geometry)
            geometries_to_split = next_geometries  # Update geometries to process in the next iteration

        # Add final geometries with attributes
        for geom in geometries_to_split:
            final_geometries.append({**hull.to_dict(), "geometry": geom})

        # Avoid duplicates in geometries for this hull
        unique_geometries = gpd.GeoDataFrame(final_geometries, crs=hulls.crs).drop_duplicates(subset="geometry")

        split_results.extend(unique_geometries.to_dict("records"))

    # Final GeoDataFrame with unique geometries and attributes
    split_gdf = gpd.GeoDataFrame(split_results, crs=hulls.crs)

    # Ensure no duplicates in the final GeoDataFrame
    split_gdf = split_gdf.drop_duplicates(subset=split_gdf.columns.to_list())

    return split_gdf


def assign_tile_best(turbines, split_polygons):
    """
    Assigns the most common 'tile_best' value from turbine points to split polygons.

    Parameters:
        turbines (GeoDataFrame): Turbine points dataset.
        split_polygons (GeoDataFrame): Split polygons dataset.

    Returns:
        GeoDataFrame: Split polygons with 'tile_best' assigned.
    """
    if "tile_best" not in turbines.columns:
        # print("[INFO] 'tile_best' field not found in turbine points. Skipping assignment.")
        split_polygons["tile_best"] = None
        return split_polygons

    # Ensure both datasets have the same CRS for spatial operations
    turbines = turbines.to_crs(split_polygons.crs)

    # Add a new column to store 'tile_best' for each polygon
    split_polygons["tile_best"] = None

    for idx, polygon in split_polygons.iterrows():
        # Select turbine points within the polygon
        turbines_within = turbines[turbines.geometry.within(polygon.geometry)]

        if not turbines_within.empty and "tile_best" in turbines_within.columns:
            # Get the most common 'tile_best' value
            most_common_tile = turbines_within["tile_best"].mode()
            if not most_common_tile.empty:
                split_polygons.at[idx, "tile_best"] = most_common_tile[0]

    return split_polygons


def merge_slivers(hulls, slivers):
    """
    Calculates the original polygon area, split polygon area, and the percentage of the original area for each split polygon.
    For polygons with less than 1% area_percentage, merges them with the first found adjacent polygon
    and removes both individual polygons, keeping only the merged result.

    Parameters:
        hulls (GeoDataFrame): The GeoDataFrame containing the original polygons before the split.
        slivers (GeoDataFrame): The GeoDataFrame containing the split polygons.

    Returns:
        GeoDataFrame: The updated GeoDataFrame with merged slivers removed and fields added.
    """
    # Ensure there is a unique identifier to match split polygons with their originals
    if "hull_id" not in hulls.columns or "hull_id" not in slivers.columns:
        raise ValueError("Both 'hulls' and 'slivers' GeoDataFrames must contain a 'hull_id' column.")

    # Add a field for the original polygon area in the hulls GeoDataFrame
    hulls["original_area"] = hulls.geometry.area

    # Add fields for split polygon area and percentage of the original area
    slivers = slivers.copy()  # Work on a copy to avoid modifying the original GeoDataFrame
    slivers["split_area"] = slivers.geometry.area
    slivers = slivers.merge(hulls[["hull_id", "original_area"]], on="hull_id", how="left")
    slivers["area_percentage"] = (slivers["split_area"] / slivers["original_area"]) * 100

    # List to hold the updated polygons
    updated_polygons = []
    processed_indices = set()  # Track processed polygons to avoid double-merging

    for idx, sliver in slivers.iterrows():
        if idx in processed_indices:
            continue  # Skip already processed polygons

        if sliver["area_percentage"] < 1:  # Check for sliver condition
            # Find adjacent polygons
            adjacent_polygons = slivers[
                slivers.geometry.touches(sliver.geometry) & ~slivers.index.isin(processed_indices)
            ]

            if not adjacent_polygons.empty:
                # Merge with the first adjacent polygon
                first_adjacent = adjacent_polygons.iloc[0]
                merged_poly = sliver.geometry.union(first_adjacent.geometry)

                # Add the merged polygon with attributes from the sliver polygon
                merged_attributes = sliver.to_dict()
                merged_attributes["geometry"] = merged_poly
                updated_polygons.append(merged_attributes)

                # Mark both polygons as processed
                processed_indices.add(idx)
                processed_indices.add(first_adjacent.name)
            else:
                # If no adjacent polygon is found, keep the sliver as is
                updated_polygons.append(sliver.to_dict())
        else:
            # If not a sliver, keep the polygon as is
            updated_polygons.append(sliver.to_dict())
            processed_indices.add(idx)

    # Create a GeoDataFrame from the updated polygons
    updated_gdf = gpd.GeoDataFrame(updated_polygons, crs=slivers.crs)

    # Ensure no duplicates in the final GeoDataFrame
    updated_gdf = updated_gdf.drop_duplicates(subset="geometry")

    updated_gdf["split_area"] = updated_gdf.geometry.area
    updated_gdf["area_percentage"] = (updated_gdf["split_area"] / updated_gdf["original_area"]) * 100

    return updated_gdf


def main():
    # Define input/output directories
    # split_dir = "data/TURBINES/"
    input_dir = "data/01_outregions/"
    output_dir = "data/02_outhulls/"

    # Define inputs
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

    datasets = ["new", "all"]

    for country in tqdm(countries, desc="Processing Countries"):
        country_output_dir = os.path.join(output_dir, country)
        os.makedirs(country_output_dir, exist_ok=True)

        # Split Convex and Concave Hulls
        # split_lines = gpd.read_file(
        #    os.path.join(split_dir, country, f"{country}_raster_tiles_poly.shp")
        # ).to_crs(CRS)
        # split_lines = split_lines.explode()  # Ensure multiline geometries are split

        for dataset in tqdm(datasets, desc=f"Processing Datasets ({country})", leave=False):
            # Input turbine points
            input_file = os.path.join(input_dir, country, f"{country}_{dataset}.gpkg")
            if not os.path.exists(input_file):
                print(f"[WARNING] File not found: {input_file}")
                continue

            turbines = gpd.read_file(input_file).to_crs(CRS)
            turbines = turbines.drop_duplicates(subset="geometry")

            # Process turbines
            turbines = clustercalculation(turbines)
            turbines.to_file(
                os.path.join(country_output_dir, f"{country}_{dataset}.gpkg"),
                layer="turbines",
            )
            # Buffer polygons
            bufferpolys = create_hulls(turbines, "buffer")

            bufferpolys.to_file(
                os.path.join(country_output_dir, f"{country}_{dataset}.gpkg"),
                layer="bufferpolys",
            )

            # Convex and Concave Hulls with IDs
            convex_hulls = create_hulls(turbines[turbines["cluster"] != -1], "convex")

            concave_hulls = create_hulls(turbines[turbines["cluster"] != -1], "concave")

            # Handle splitting and merging slivers for concave polygons
            # split_concave = split_hulls(concave_hulls, split_lines)
            # split_concave = assign_tile_best(turbines, split_concave)

            # Merge slivers in concave hulls
            # merged_concave = merge_slivers(hulls=concave_hulls, slivers=split_concave)
            concave_hulls.to_file(
                os.path.join(country_output_dir, f"{country}_{dataset}.gpkg"),
                layer="concavehulls",
            )

            # Handle splitting and merging slivers for convex polygons
            # split_convex = split_hulls(convex_hulls, split_lines)
            # split_convex = assign_tile_best(turbines, split_convex)

            # Merge slivers in convex hulls
            # merged_convex = merge_slivers(hulls=convex_hulls, slivers=split_convex)
            convex_hulls.to_file(
                os.path.join(country_output_dir, f"{country}_{dataset}.gpkg"),
                layer="convexhulls",
            )


main()
