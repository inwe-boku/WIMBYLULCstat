import argparse
import os

import geopandas as gpd
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration Variables (can be overridden via command-line arguments)
# ----------------------------------------------------------------------------
INPUT_DIR = "data/TURBINES"
SECOND_DATASETS_FOLDER = "data/REGIONS"
OUTPUT_DIR = "data/01_outregions"
DATASETS = ["new", "all", "old"]
TURBINE_LAYER = "turbines"
MAX_DISTANCE = 5000  # meters for nearest-join in second datasets
OLD_DISTANCE = 99  # meters proximity to consider turbines as 'old'
SECOND_DATASETS_CONFIG = {
    "NUTS.gpkg": ["NUTS1_CODE", "NUTS2_CODE", "NUTS3_CODE", "NUTS3_NAME"],
    "WWF_Ecos.gpkg": ["WWFEcos_code", "WWFEcos_name", "WWFEcos_realm", "WWFEcos_biome"],
    "EEA_Biogeoregions2016.gpkg": ["EEABGR_code", "EEABGR_sname", "EEABGR_name"],
    "EC_ENSv8.gpkg": ["EnS_code", "EnS_name", "EnZ_code", "EnZ_name"],
}


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with attribute 'countries'
    """
    parser = argparse.ArgumentParser(
        description="Process turbine shapefiles by joining with region datasets and extracting 'old' turbines."
    )
    parser.add_argument(
        "--countries",
        type=str,
        help="Comma-separated list of country folder names to process (overrides auto-detection).",
    )
    return parser.parse_args()


def get_countries(input_dir, override_list=None):
    """
    Determine the list of countries to process.

    Args:
        input_dir (str): Path to the turbines input directory.
        override_list (list[str] or None): List of countries provided via CLI.

    Returns:
        list[str]: Sorted list of country folder names.
    """
    if override_list:
        # Use only the user-specified subset
        return [c.strip() for c in override_list]
    # Auto-detect by listing subdirectories in input_dir
    try:
        entries = os.listdir(input_dir)
    except FileNotFoundError:
        return []
    countries = [d for d in entries if os.path.isdir(os.path.join(input_dir, d))]
    return sorted(countries)


def process_single_dataset(country, dataset, second_datasets_folder, output_dir):
    """
    Process a single dataset type ('new', 'all', etc.) for a country.

    Args:
        country (str): Country folder name.
        dataset (str): Dataset suffix ('new', 'all', 'old').
        second_datasets_folder (str): Path to region datasets.
        output_dir (str): Base path to write outputs.

    Returns:
        bool: True if processing succeeded (shapefile existed), False otherwise.
    """
    shp_file = os.path.join(INPUT_DIR, country, f"{country}_{dataset}.shp")
    if not os.path.exists(shp_file):
        return False

    try:
        gdf = gpd.read_file(shp_file)
    except Exception:
        return False

    # Deduplicate and index
    gdf = gdf.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    gdf["fid"] = range(1, len(gdf) + 1)
    result = gdf

    # Join with each secondary dataset
    for fname, fields in SECOND_DATASETS_CONFIG.items():
        fpath = os.path.join(second_datasets_folder, fname)
        if not os.path.exists(fpath):
            continue
        regions = gpd.read_file(fpath).to_crs(result.crs)
        missing = [f for f in fields if f not in regions.columns]
        if missing:
            continue
        regions = regions[["geometry"] + fields]

        # Intersect
        join = gpd.sjoin(result, regions, how="left", predicate="intersects")
        join_cols = [c for c in fields if c in join.columns]
        unmatched = join[join[join_cols].isna().all(axis=1)]
        if not unmatched.empty:
            clean = unmatched.drop(columns=join_cols + ["index_right"], errors="ignore")
            nearest = gpd.sjoin_nearest(
                clean,
                regions,
                how="left",
                max_distance=MAX_DISTANCE,
                distance_col=None,
            )
            nearest = nearest.loc[~nearest.index.duplicated(keep="first")]
            for col in fields:
                join.loc[nearest.index, col] = nearest[col]

        # Prepare for next iteration
        result = join.drop(columns={"index_right", "fid"} & set(join.columns), errors="ignore")
        result = result[~result.index.duplicated(keep="first")].reset_index(drop=True)

    # Write output
    country_out = os.path.join(output_dir, country)
    os.makedirs(country_out, exist_ok=True)
    out_path = os.path.join(country_out, f"{country}_{dataset}.gpkg")
    result.to_file(out_path, layer=TURBINE_LAYER)
    return True


def create_old_turbines(country, output_dir):
    """
    Generate the 'old' turbine dataset by removing 'all' turbines
    within OLD_DISTANCE of any 'new' turbine.

    Args:
        country (str): Country folder name.
        output_dir (str): Base path where outputs reside.
    """
    country_out = os.path.join(output_dir, country)
    all_fp = os.path.join(country_out, f"{country}_all.gpkg")
    new_fp = os.path.join(country_out, f"{country}_new.gpkg")
    old_fp = os.path.join(country_out, f"{country}_old.gpkg")

    if not (os.path.exists(all_fp) and os.path.exists(new_fp)):
        return

    gdf_all = gpd.read_file(all_fp)
    gdf_new = gpd.read_file(new_fp)
    if gdf_all.empty or gdf_new.empty:
        gdf_old = gdf_all
    else:
        joined = gpd.sjoin_nearest(
            gdf_all,
            gdf_new[[gdf_new.geometry.name]],
            how="left",
            max_distance=OLD_DISTANCE,
        )
        to_remove = joined[~joined["index_right"].isna()].index
        gdf_old = joined.drop(index=to_remove)
        gdf_old = gdf_old.drop(columns=["index_right"], errors="ignore")

    gdf_old.to_file(old_fp, layer=TURBINE_LAYER)


def process_turbines(countries, second_datasets_folder, output_dir):
    """
    Main processing function: handles all countries and datasets.

    Args:
        countries (list[str]): List of country folder names.
        second_datasets_folder (str): Path to region datasets.
        output_dir (str): Base path to write outputs.
    """
    created = {}
    for country in tqdm(countries, desc="Countries"):
        successes = []
        for ds in DATASETS:
            ok = process_single_dataset(country, ds, second_datasets_folder, output_dir)
            if ok:
                successes.append(ds)
        created[country] = successes

        # Create 'old' only if both 'new' and 'all' succeeded
        if "new" in successes and "all" in successes:
            create_old_turbines(country, output_dir)


def main():
    """
    Entry point: parse arguments, determine countries, and initiate processing.
    """
    args = parse_args()
    override = args.countries.split(",") if args.countries else None
    countries = get_countries(INPUT_DIR, override)

    if not countries:
        print(f"No countries found in '{INPUT_DIR}'. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_turbines(countries, SECOND_DATASETS_FOLDER, OUTPUT_DIR)


if __name__ == "__main__":
    main()
