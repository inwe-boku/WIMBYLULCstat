import os

import geopandas as gpd
from tqdm import tqdm

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

datasets = ["new", "all", "old"]
turbine_layer = "turbines"

# Path to the folder containing second datasets
second_datasets_folder = "data/REGIONS"

# Configuration for second datasets and fields to map
second_datasets_config = {
    "NUTS.gpkg": ["NUTS1_CODE", "NUTS2_CODE", "NUTS3_CODE", "NUTS3_NAME"],
    "WWF_Ecos.gpkg": ["WWFEcos_code", "WWFEcos_name", "WWFEcos_realm", "WWFEcos_biome"],
    "EEA_Biogeoregions2016.gpkg": ["EEABGR_code", "EEABGR_sname", "EEABGR_name"],
    "EC_ENSv8.gpkg": ["EnS_code", "EnS_name", "EnZ_code", "EnZ_name"],
}

output_dir = "data/01_outregions"  # Directory to store processed datasets
os.makedirs(output_dir, exist_ok=True)

# Specify a distance threshold for the second dataset nearest-join (in CRS units, typically meters):
MAX_DISTANCE = 5000

# Specify the distance threshold to define "old" turbines (e.g., 250 meters)
OLD_DISTANCE = 99


def process_turbines():
    for country in tqdm(countries, desc="Processing Countries"):
        # Ensure the country's output directory exists
        country_output_dir = os.path.join(output_dir, country)
        os.makedirs(country_output_dir, exist_ok=True)

        # We will store which datasets we actually processed successfully
        created_files = []

        for dataset in tqdm(datasets, desc=f"Processing datasets in {country}", leave=False):
            # Construct the file name
            shp_file = f"data/TURBINES/{country}/{country}_{dataset}.shp"
            if not os.path.exists(shp_file):
                continue  # Skip if file not found

            try:
                # Read the turbine file
                turbines_gdf = gpd.read_file(shp_file)
            except Exception:
                continue  # Skip if layer reading fails

            # Remove duplicate features immediately after reading the input
            turbines_gdf = turbines_gdf.drop_duplicates(subset=["geometry"], keep="first")
            turbines_gdf = turbines_gdf.reset_index(drop=True)
            turbines_gdf["fid"] = range(1, len(turbines_gdf) + 1)

            # Result starts as just the turbines
            result_gdf = turbines_gdf

            # Process second datasets
            for second_file_name, fields_to_transfer in second_datasets_config.items():
                second_file_path = os.path.join(second_datasets_folder, second_file_name)
                if not os.path.exists(second_file_path):
                    # Skip if second dataset file not found
                    continue

                # Read polygons
                second_gdf = gpd.read_file(second_file_path)

                # Ensure both datasets have the same CRS
                if result_gdf.crs != second_gdf.crs:
                    second_gdf = second_gdf.to_crs(result_gdf.crs)

                # Check fields
                missing_fields = [field for field in fields_to_transfer if field not in second_gdf.columns]
                if missing_fields:
                    # Skip if required fields are missing in second dataset
                    continue

                # We only need geometry + the attribute fields
                second_gdf = second_gdf[["geometry"] + fields_to_transfer]

                # 1) Intersect join
                intersect_join = gpd.sjoin(result_gdf, second_gdf, how="left", predicate="intersects")

                # Keep track of which columns came from second_gdf
                join_cols = [col for col in intersect_join.columns if col in fields_to_transfer]

                # 2) For unmatched points, do a nearest join within MAX_DISTANCE
                unmatched = intersect_join[intersect_join[join_cols].isna().all(axis=1)].copy()

                if len(unmatched) > 0:
                    # Drop join fields and any conflicting "index_right" to avoid conflicts
                    unmatched_clean = unmatched.drop(columns=join_cols + ["index_right"], errors="ignore")

                    # Perform the nearest join
                    nearest_join = gpd.sjoin_nearest(
                        unmatched_clean,
                        second_gdf,
                        how="left",
                        max_distance=MAX_DISTANCE,
                        distance_col=None,
                    )

                    # Remove duplicate matches, keeping only the first match for each turbine
                    nearest_join = nearest_join.loc[~nearest_join.index.duplicated(keep="first")]

                    # Replace NaNs in the original join with the values from nearest_join
                    for col in fields_to_transfer:
                        intersect_join.loc[nearest_join.index, col] = nearest_join[col]

                # Update result_gdf with final attributes after intersect + nearest
                # Drop geometry columns we don't need and index_right/fid if present
                columns_to_drop = {"index_right", "fid"}
                result_gdf = intersect_join.drop(columns=columns_to_drop & set(intersect_join.columns), errors="ignore")

                # Remove duplicated index if any
                result_gdf = result_gdf.loc[~result_gdf.index.duplicated(keep="first")]
                result_gdf = result_gdf.reset_index(drop=True)

            # Save the processed turbine points
            output_path = os.path.join(country_output_dir, f"{country}_{dataset}.gpkg")
            result_gdf.to_file(output_path, layer=turbine_layer)
            created_files.append(dataset)

        # ----------------------------------------------------------------------
        # Create the "old" dataset for this country: remove any "all" turbine
        # that is within OLD_DISTANCE meters of a "new" turbine
        # ----------------------------------------------------------------------
        # Only do this if BOTH all and new data were processed successfully
        if "all" in created_files and "new" in created_files:
            all_path = os.path.join(country_output_dir, f"{country}_all.gpkg")
            new_path = os.path.join(country_output_dir, f"{country}_new.gpkg")
            if os.path.exists(all_path) and os.path.exists(new_path):
                gdf_all = gpd.read_file(all_path)
                gdf_new = gpd.read_file(new_path)

                if not gdf_all.empty and not gdf_new.empty:
                    # Spatial nearest join: link every turbine in gdf_all to the closest
                    # turbine in gdf_new, only if within OLD_DISTANCE.
                    joined = gpd.sjoin_nearest(
                        gdf_all,
                        gdf_new[[gdf_new.geometry.name]],
                        how="left",
                        max_distance=OLD_DISTANCE,
                    )

                    # If "index_right" is NOT NaN => we matched a "new" turbine within OLD_DISTANCE
                    to_remove = joined[~joined["index_right"].isna()].index
                    gdf_old = joined.loc[~joined.index.isin(to_remove)]

                    # Drop sjoin columns we no longer need
                    if "index_right" in gdf_old.columns:
                        gdf_old.drop(columns=["index_right"], inplace=True)
                else:
                    # If one is empty, old = all
                    gdf_old = gdf_all

                old_path = os.path.join(country_output_dir, f"{country}_old.gpkg")
                gdf_old.to_file(old_path, layer=turbine_layer)
            # else: no action needed if files don't exist


if __name__ == "__main__":
    process_turbines()
