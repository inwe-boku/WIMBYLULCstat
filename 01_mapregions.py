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

datasets = ["new", "all"]
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


# Function to process turbine points
def process_turbines():
    for country in tqdm(countries, desc="Processing Countries"):
        # Ensure the country's output directory exists
        country_output_dir = os.path.join(output_dir, country)
        os.makedirs(country_output_dir, exist_ok=True)

        for dataset in tqdm(datasets, desc=f"Processing datasets in {country}", leave=False):
            # Construct the GeoPackage file name
            gpkg_file = f"data/TURBINES/{country}/{country}_{dataset}.shp"
            if not os.path.exists(gpkg_file):
                continue  # Skip if file not found

            try:
                # Read the turbine layer
                turbines_gdf = gpd.read_file(gpkg_file)
            except Exception:
                continue  # Skip if layer reading fails

            result_gdf = turbines_gdf

            # Process second datasets
            for second_file_name, fields_to_transfer in second_datasets_config.items():
                second_file_path = os.path.join(second_datasets_folder, second_file_name)
                if not os.path.exists(second_file_path):
                    continue  # Skip if file not found

                second_gdf = gpd.read_file(second_file_path)

                # Ensure both datasets have the same CRS
                if result_gdf.crs != second_gdf.crs:
                    second_gdf = second_gdf.to_crs(result_gdf.crs)

                missing_fields = [field for field in fields_to_transfer if field not in second_gdf.columns]
                if missing_fields:
                    continue  # Skip if required fields are missing

                second_gdf = second_gdf[["geometry"] + fields_to_transfer]
                result_gdf = gpd.sjoin(result_gdf, second_gdf, how="left", predicate="intersects")
                # Drop the geometry column from second_gdf, keeping only the attribute columns
                result_gdf = result_gdf.drop(columns=["index_right", "fid"], errors="ignore")
                result_gdf = result_gdf.loc[~result_gdf.index.duplicated(keep="first")]
                result_gdf = result_gdf.reset_index(drop=True)

            # Save the processed turbine points
            output_path = os.path.join(country_output_dir, f"{country}_{dataset}.gpkg")
            result_gdf.to_file(output_path, layer=turbine_layer)


if __name__ == "__main__":
    process_turbines()
