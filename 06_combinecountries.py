import glob
import os

import fiona
import geopandas as gpd
import pandas as pd


def read_multi_layer_gpkg(file_path):
    """
    Reads all layers from a GPKG and returns a dict: {layer_name: GeoDataFrame}.
    """
    gdf_dict = {}
    layers = fiona.listlayers(file_path)
    for layer in layers:
        try:
            gdf = gpd.read_file(file_path, layer=layer)
            gdf_dict[layer] = gdf
        except Exception as e:
            print(f"Error reading layer='{layer}' in '{file_path}': {e}")
    return gdf_dict


def write_multi_layer_gpkg(gdf_dict, out_gpkg_path):
    """
    Write a dictionary of {layer_name: GeoDataFrame} to a single multi-layer GPKG file.
    Overwrites if it already exists.
    """
    if os.path.exists(out_gpkg_path):
        os.remove(out_gpkg_path)

    for layer_name, gdf in gdf_dict.items():
        if gdf.empty:
            continue
        gdf.to_file(out_gpkg_path, layer=layer_name, driver="GPKG")


def save_feather_files(gdf_dict, out_dir, prefix=""):
    """
    For each layer in gdf_dict, save the attribute table (dropped geometry) as a Feather file.
    The filename is '{prefix}{layer_name}.feather'.
    """
    os.makedirs(out_dir, exist_ok=True)

    for layer_name, gdf in gdf_dict.items():
        # Drop geometry
        df_no_geom = gdf.drop(columns="geometry", errors="ignore").fillna(0)
        df_no_geom = df_no_geom[sorted(df_no_geom.columns)]

        # Convert object columns to string
        for col in df_no_geom.select_dtypes(include=["object"]).columns:
            df_no_geom[col] = df_no_geom[col].astype(str)

        # Write to Feather
        feather_file = os.path.join(out_dir, f"{prefix}{layer_name}.feather")
        df_no_geom.reset_index(drop=True).to_feather(feather_file, compression="zstd")


def get_old_data(all_gdf, new_gdf, radius_for_points=0.0):
    """
    Return 'old' features from all_gdf that are not considered 'new' by:
      - Points: within radius_for_points => considered 'new'.
      - Polygons/others: intersects => considered 'new'.
    """
    if all_gdf.empty:
        return all_gdf

    # We assume geometry type is consistent for each layer
    geom_type = all_gdf.geom_type.iloc[0]

    if geom_type == "Point":
        if new_gdf.empty:
            # No new points => all are old
            return all_gdf

        # Mark any point as 'new' if it's within radius_for_points of ANY new point
        def is_within_radius(all_feat):
            return new_gdf.distance(all_feat.geometry).min() <= radius_for_points

        mask = all_gdf.apply(is_within_radius, axis=1)
        old_gdf = all_gdf.loc[~mask]

    else:
        # Polygons or lines => "new" if intersects any geometry in new_gdf
        if new_gdf.empty:
            return all_gdf

        def intersects_new(all_feat):
            return new_gdf.intersects(all_feat.geometry).any()

        mask = all_gdf.apply(intersects_new, axis=1)
        old_gdf = all_gdf.loc[~mask]

    return old_gdf


def process_country(new_file, all_file, output_dir, radius_for_points=0.0, verbose=False):
    """
    For a single country, read *new.gpkg and *all.gpkg:
      1) read each into dict {layer_name: gdf}
      2) compute 'old' for each layer
      3) write 'new', 'all', 'old' each into a separate multi-layer GPKG
      4) also write each layer's attribute table as Feather

    Returns 3 dicts: (country_new_dict, country_all_dict, country_old_dict).
    """
    country_name = os.path.basename(new_file).split("_")[0]

    # Read all layers
    new_dict = read_multi_layer_gpkg(new_file)
    all_dict = read_multi_layer_gpkg(all_file)
    old_dict = {}

    for layer_name, all_gdf in all_dict.items():
        new_gdf = new_dict.get(layer_name, gpd.GeoDataFrame(columns=["geometry"]))
        old_gdf = get_old_data(all_gdf, new_gdf, radius_for_points)
        old_dict[layer_name] = old_gdf

    # Write out multi-layer GPKG
    country_out_dir = os.path.join(output_dir, country_name)
    os.makedirs(country_out_dir, exist_ok=True)

    new_gpkg = os.path.join(country_out_dir, f"{country_name}_new.gpkg")
    all_gpkg = os.path.join(country_out_dir, f"{country_name}_all.gpkg")
    old_gpkg = os.path.join(country_out_dir, f"{country_name}_old.gpkg")

    write_multi_layer_gpkg(new_dict, new_gpkg)
    write_multi_layer_gpkg(all_dict, all_gpkg)
    write_multi_layer_gpkg(old_dict, old_gpkg)

    # Also save each layer's attribute table as Feather
    # (prefix them with new_, all_, old_)
    save_feather_files(new_dict, country_out_dir, prefix="new_")
    save_feather_files(all_dict, country_out_dir, prefix="all_")
    save_feather_files(old_dict, country_out_dir, prefix="old_")

    if verbose:
        print(f"Processed country={country_name}, wrote {new_gpkg}, {all_gpkg}, {old_gpkg}")

    return new_dict, all_dict, old_dict


def combine_europe_data(europe_out_dir, all_countries_new, all_countries_all, all_countries_old, verbose=False):
    """
    Combine all countries' new/all/old data into 3 Europe-wide multi-layer GPKGs + Feather:
      - europe_new.gpkg
      - europe_all.gpkg
      - europe_old.gpkg
    each a multi-layer GPKG across all layers seen in the union of countries.
    """
    os.makedirs(europe_out_dir, exist_ok=True)

    # For each "type" (new/all/old), gather by layer
    # e.g. europe_new_dict[layer_name] = [gdfs from each country]
    # Then we write one multi-layer file.

    def merge_gdfs_by_layer(list_of_dicts):
        """
        list_of_dicts is a list of {layer: GeoDataFrame} from each country.
        We produce a new dict {layer: merged_gdf} with all countries merged for each layer.
        """
        merged_dict = {}
        # Collect all possible layer names
        all_layer_names = set()
        for dct in list_of_dicts:
            all_layer_names.update(dct.keys())

        for layer_name in all_layer_names:
            # gather all country gdfs for this layer
            part_gdfs = []
            for dct in list_of_dicts:
                if layer_name in dct:
                    part_gdfs.append(dct[layer_name])
            if part_gdfs:
                merged_dict[layer_name] = gpd.GeoDataFrame(pd.concat(part_gdfs, ignore_index=True))
            else:
                merged_dict[layer_name] = gpd.GeoDataFrame(columns=["geometry"])
        return merged_dict

    # Merge them
    europe_new = merge_gdfs_by_layer(all_countries_new)
    europe_all = merge_gdfs_by_layer(all_countries_all)
    europe_old = merge_gdfs_by_layer(all_countries_old)

    # Write multi-layer GPKG
    europe_new_gpkg = os.path.join(europe_out_dir, "europe_new.gpkg")
    europe_all_gpkg = os.path.join(europe_out_dir, "europe_all.gpkg")
    europe_old_gpkg = os.path.join(europe_out_dir, "europe_old.gpkg")

    write_multi_layer_gpkg(europe_new, europe_new_gpkg)
    write_multi_layer_gpkg(europe_all, europe_all_gpkg)
    write_multi_layer_gpkg(europe_old, europe_old_gpkg)

    # Write Feather
    save_feather_files(europe_new, europe_out_dir, prefix="europe_new_")
    save_feather_files(europe_all, europe_out_dir, prefix="europe_all_")
    save_feather_files(europe_old, europe_out_dir, prefix="europe_old_")

    if verbose:
        print(f"Created {europe_new_gpkg}, {europe_all_gpkg}, {europe_old_gpkg}")


def main():
    # Adjust these paths as needed
    input_directory = "data/05_stat/gpkg/"
    output_directory = "data/06_combined/"
    radius_for_points = 0.0  # 0 => exact match for Points

    # We expect pairs: <country>_new.gpkg, <country>_all.gpkg
    new_files = sorted(glob.glob(os.path.join(input_directory, "*new.gpkg")))
    all_files = sorted(glob.glob(os.path.join(input_directory, "*all.gpkg")))

    # We will store the dictionaries from each country to combine for Europe
    all_countries_new_dicts = []
    all_countries_all_dicts = []
    all_countries_old_dicts = []

    # Pair up the new/all GPKGs by matching country name
    # (Assuming the same prefix and 1:1 mapping)
    matched_pairs = []
    for nf in new_files:
        country = os.path.basename(nf).split("_")[0]
        # Try to find a matching all-file for that country
        # E.g. BEL_new.gpkg => BEL_all.gpkg
        candidate = os.path.join(input_directory, f"{country}_all.gpkg")
        if os.path.exists(candidate):
            matched_pairs.append((nf, candidate))
        else:
            print(f"No matching '*all.gpkg' for '{nf}'")

    for new_file, all_file in matched_pairs:
        # Process the country, get (new_dict, all_dict, old_dict)
        new_d, all_d, old_d = process_country(
            new_file, all_file, output_dir=output_directory, radius_for_points=radius_for_points, verbose=True
        )
        all_countries_new_dicts.append(new_d)
        all_countries_all_dicts.append(all_d)
        all_countries_old_dicts.append(old_d)

    # After all countries are done, combine each "type" (new, all, old) into Europe-level GPKGs
    europe_out_dir = os.path.join(output_directory, "europe")
    combine_europe_data(
        europe_out_dir=europe_out_dir,
        all_countries_new=all_countries_new_dicts,
        all_countries_all=all_countries_all_dicts,
        all_countries_old=all_countries_old_dicts,
        verbose=True,
    )


if __name__ == "__main__":
    main()
