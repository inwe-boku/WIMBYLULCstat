import glob
import os

import fiona
import geopandas as gpd
import pandas as pd


def save_dataframes(output_dir, gdf, layer_name):
    """Save the dataframe as GPKG, CSV, and Feather files."""
    os.makedirs(output_dir, exist_ok=True)
    csv_filepath = os.path.join(output_dir, f"{layer_name}.csv")
    feather_filepath = os.path.join(output_dir, f"{layer_name}.feather")
    gpkg_filepath = os.path.join(output_dir, f"{layer_name}.gpkg")

    gdf.to_file(gpkg_filepath, layer=layer_name, driver="GPKG")
    df_no_geom = gdf.drop(columns="geometry", errors="ignore").fillna(0)
    df_no_geom = df_no_geom[sorted(df_no_geom.columns)]
    df_no_geom.to_csv(csv_filepath, index=False)
    for col in df_no_geom.select_dtypes(include=["object"]).columns:
        df_no_geom[col] = df_no_geom[col].astype(str)
    df_no_geom.reset_index(drop=True).to_feather(feather_filepath, compression="zstd")


def combine_gpkg_files_in_directory(directory, file_pattern, output_prefix, verbose=True):
    if verbose:
        print(f"Searching for GeoPackage files in directory: {directory} with pattern: {file_pattern}")

    gpkg_files = glob.glob(os.path.join(directory, file_pattern))

    if not gpkg_files:
        raise ValueError("No GeoPackage files found in the specified directory.")

    if verbose:
        print(f"Found {len(gpkg_files)} GeoPackage file(s).")

    combined_layers = {}

    for file in gpkg_files:
        country_name = os.path.basename(file).split("_")[0]
        output_dir = os.path.join(directory, country_name)
        layers = fiona.listlayers(file)
        for layer in layers:
            try:
                gdf = gpd.read_file(file, layer=layer)
                save_dataframes(output_dir, gdf, f"{output_prefix}_{layer}")
                if layer not in combined_layers:
                    combined_layers[layer] = []
                combined_layers[layer].append(gdf)
                if verbose:
                    print(f"  Successfully wrote layer: {layer} for {country_name}")
            except Exception as e:
                print(f"  Failed to read/write layer: {layer} for {country_name} - {e}")

    europe_dir = os.path.join(directory, "europe")
    for layer, gdfs in combined_layers.items():
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        save_dataframes(europe_dir, combined_gdf, f"europe_{output_prefix}_{layer}")


def process_difference(directory, verbose=True):
    new_files = glob.glob(os.path.join(directory, "*new.gpkg"))
    all_files = glob.glob(os.path.join(directory, "*all.gpkg"))

    for new_file, all_file in zip(new_files, all_files):
        country_name = os.path.basename(new_file).split("_")[0]
        layers = fiona.listlayers(new_file)
        for layer in layers:
            try:
                new_gdf = gpd.read_file(new_file, layer=layer)
                all_gdf = gpd.read_file(all_file, layer=layer)
                touch_mask = all_gdf.apply(lambda x: new_gdf.intersects(x.geometry).any(), axis=1)
                difference = all_gdf.loc[~touch_mask]

                output_dir = os.path.join(directory, country_name)
                save_dataframes(output_dir, difference, f"old_{layer}")
                if verbose:
                    print(f"  Saved difference for {country_name} - Layer: {layer}")
            except Exception as e:
                print(f"  Error processing difference for {country_name} - Layer: {layer} - {e}")

    europe_dir = os.path.join(directory, "europe")
    for layer in layers:
        try:
            combined_diffs = []
            for new_file, all_file in zip(new_files, all_files):
                new_gdf = gpd.read_file(new_file, layer=layer)
                all_gdf = gpd.read_file(all_file, layer=layer)
                touch_mask = all_gdf.apply(lambda x: new_gdf.intersects(x.geometry).any(), axis=1)
                difference = all_gdf.loc[~touch_mask]
                combined_diffs.append(difference)
            europe_diff = gpd.GeoDataFrame(pd.concat(combined_diffs, ignore_index=True))
            save_dataframes(europe_dir, europe_diff, f"europe_old_{layer}")
        except Exception as e:
            print(f"  Error processing Europe difference - Layer: {layer} - {e}")


def main():
    directory_path = "data/06_comb"
    combine_gpkg_files_in_directory(directory_path, "*new.gpkg", "new")
    combine_gpkg_files_in_directory(directory_path, "*all.gpkg", "all")
    process_difference(directory_path)


if __name__ == "__main__":
    main()
