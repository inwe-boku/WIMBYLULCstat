import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

# EPSG definition
# crs equal area Europe: 3035
ccrs = "epsg:3035"

clc_datasets = ["clc100", "clc10"]

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

countries = ["poland"]

gpkg_path = "data/04_clc"  # Replace with your GeoPackage file path
out_path = "data/05_stat"
poly_layers = ["bufferpolys", "concavehulls", "convexhulls"]  # Layers to process

resultfields = ["area", "carea", "careap", "ashare"]

clc_file = "data/CLC/CLC.xlsx"

datasets = ["new", "all"]

turbinecoldict = {
    "hubheight": "mean",
    "power": "mean",
    "diameter": "mean",
    "date": "mode",
    "manufactur": "mode",
}


def copy_nearest_columns(source_gdf, target_gdf, source_columns, max_distance=500, target_crs="EPSG:3035"):
    """
    Copies specified columns from the nearest feature in the source GeoDataFrame
    to the target GeoDataFrame using BallTree for nearest neighbor search.

    Parameters:
        source_gdf (GeoDataFrame): The source GeoDataFrame containing the data to copy.
        target_gdf (GeoDataFrame): The target GeoDataFrame where the data will be copied.
        source_columns (list): List of column names to copy from the source GeoDataFrame.
        target_crs (str): CRS to use for spatial operations. Default is 'EPSG:4326'.

    Returns:
        GeoDataFrame: Updated target GeoDataFrame with added columns from the nearest source features.
    """
    # Ensure both GeoDataFrames have the same CRS
    source_gdf = source_gdf.to_crs(target_crs)
    target_gdf = target_gdf.to_crs(target_crs)

    # Extract coordinates as numpy arrays
    source_coords = np.array(list(zip(source_gdf.geometry.x, source_gdf.geometry.y)))
    target_coords = np.array(list(zip(target_gdf.geometry.x, target_gdf.geometry.y)))

    # Build a BallTree for nearest neighbor search
    tree = BallTree(source_coords, metric="haversine")

    # Query the nearest source for each target
    distances, indices = tree.query(target_coords, k=1)

    # Convert distances from radians to meters (assuming WGS84 CRS)
    distances = distances * 6371000  # Earth's radius in meters

    valid_mask = distances.flatten() < max_distance

    # Copy the specified columns from the nearest source features
    for column in source_columns:
        target_gdf[column] = None
        target_gdf.loc[valid_mask, column] = source_gdf.iloc[indices.flatten()[valid_mask]][column].values
    return target_gdf


def points_within_polygons(points_gdf, polygons_gdf, agg_dict):
    """
    Matches points to polygons, aggregates selected columns with specified methods,
    and adds them to the polygon GeoDataFrame along with a count of points.

    Parameters:
    - points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries.
    - polygons_gdf (gpd.GeoDataFrame): GeoDataFrame containing polygon geometries.
    - agg_dict (dict): Dictionary where keys are column names from points_gdf and values are aggregation methods.

    Returns:
    - gpd.GeoDataFrame: Updated polygons GeoDataFrame with new aggregated columns and a point count column `tcount`.
    """

    # Define custom aggregation methods
    def mode(series):
        return series.mode().iloc[0] if not series.mode().empty else None

    # Map method names to actual functions
    aggregation_methods = {
        "mean": "mean",
        "sum": "sum",
        "median": "median",
        "first": "first",
        "last": "last",
        "mode": mode,
    }

    # Validate aggregation methods in the dictionary
    agg_func_dict = {
        field: aggregation_methods[method] if method in aggregation_methods else method
        for field, method in agg_dict.items()
    }

    # Spatial join to find which points fall within each polygon
    joined = gpd.sjoin(points_gdf, polygons_gdf, how="inner", predicate="within")

    # Determine the correct column name for merging
    index_col = "index_right" if "index_right" in joined.columns else "index_right0"

    # Group by the correct index column and count occurrences
    tcount = joined.groupby(index_col).size().reset_index(name="tcount")

    # Aggregate data using the correct index column
    aggregated = joined.groupby(index_col).agg(agg_func_dict).reset_index().rename(columns={index_col: "polygon_index"})

    # Merge and clean up using the determined index column
    aggregated = aggregated.merge(tcount, left_on="polygon_index", right_on=index_col, how="left")
    polygons_gdf = polygons_gdf.reset_index().merge(aggregated, left_index=True, right_on="polygon_index", how="left")
    polygons_gdf.drop(columns=["polygon_index", index_col], inplace=True)

    return polygons_gdf


def main():
    # Set the GeoPackage file path and layers

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "xls"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "gpkg"), exist_ok=True)

    bindex = resultfields * len(poly_layers)  # Corrected: Multiply by the number of poly_layers
    tindex = [
        item for item in poly_layers for _ in range(len(resultfields))
    ]  # Corrected: range should be len(resultfields)

    clcdata = {}
    print("Loading CLC data...")
    for clcd in clc_datasets:
        try:
            clcxls = pd.read_excel(clc_file, sheet_name=clcd)
            # Ensure 'Category' and 'CatName' exist before attempting group by/ffill
            if "Category" in clcxls.columns and "CatName" in clcxls.columns:
                clcxls["CatName"] = clcxls.groupby("Category")["CatName"].transform(lambda x: x.ffill())
            else:
                print(f"Warning: 'Category' or 'CatName' not found in sheet {clcd}. Skipping ffill.")
            clcdata[clcd] = clcxls
        except Exception as e:
            print(f"Error loading sheet {clcd} from {clc_file}: {e}")
            # Decide how to handle: skip, exit, use empty dataframe?
            clcdata[clcd] = pd.DataFrame()  # Example: use empty dataframe

    # Define multi-index templates based on loaded CLC data structure
    # Assuming 'Value', 'Code', 'Name', 'Category', 'CatName' exist in clc100 sheet
    base_cols_100 = (
        ["Value", "Code", "Name", "Category", "CatName"] if "clc100" in clcdata and not clcdata["clc100"].empty else []
    )
    clc100bindex = base_cols_100 + bindex
    clc100tindex = [""] * len(base_cols_100) + tindex

    # Assuming 'Value', 'Name', 'Category', 'CatName' exist in clc10 sheet
    base_cols_10 = ["Value", "Name", "Category", "CatName"] if "clc10" in clcdata and not clcdata["clc10"].empty else []
    clc10bindex = base_cols_10 + bindex
    clc10tindex = [""] * len(base_cols_10) + tindex

    dtuosm_turbines = {}
    print("Loading DTU/OSM turbine data...")
    for newall in datasets:
        turbine_DTU_OSM_file = f"data/DTU_OSM/DTU_OSM_{newall}.gpkg"  # Use f-string
        try:
            turbines = gpd.read_file(turbine_DTU_OSM_file)
            dtuosm_turbines[newall] = turbines
        except Exception as e:
            print(f"Error loading DTU/OSM file {turbine_DTU_OSM_file}: {e}")
            # Decide how to handle: skip, exit?
            dtuosm_turbines[newall] = gpd.GeoDataFrame()  # Example: use empty geodataframe

    # --- Outer loop with progress bar for countries ---
    for country in tqdm(countries, desc="Processing Countries"):
        # --- Inner loop with progress bar for datasets (newall) ---
        # leave=False removes the inner bar once completed for the current country
        for newall in tqdm(datasets, desc=f"Processing '{country}' Datasets", leave=False):
            file_path = os.path.join(gpkg_path, country, f"{country}_{newall}.gpkg")
            gpkgfile = os.path.join(out_path, "gpkg", f"{country}_{newall}.gpkg")
            xlsfile = os.path.join(out_path, "xls", f"{country}_{newall}.xlsx")

            try:
                # Load turbines for the specific country/dataset
                turbines = gpd.read_file(file_path, layer="turbines")

                # Check if corresponding DTU/OSM data exists before copying
                if newall in dtuosm_turbines and not dtuosm_turbines[newall].empty:
                    turbines = copy_nearest_columns(dtuosm_turbines[newall], turbines, list(turbinecoldict.keys()))
                else:
                    print(f"Warning: No DTU/OSM data found or loaded for {newall}. Skipping column copy for {country}.")

                # Save updated turbines
                turbines.to_file(gpkgfile, layer="turbines", driver="GPKG")

                # Process polygon layers and write to Excel
                with pd.ExcelWriter(xlsfile) as writer:
                    for clcd in clc_datasets:
                        # Check if clc data for this type was loaded successfully
                        if clcd not in clcdata or clcdata[clcd].empty:
                            print(f"Skipping CLC type {clcd} for {country}/{newall} due to loading errors.")
                            continue

                        clcres = pd.DataFrame()
                        multi_index = None  # Initialize multi_index

                        # Set up MultiIndex based on clcd type
                        if clcd == "clc100":
                            # Check if required columns exist for index creation
                            if len(clc100tindex) == len(clc100bindex) and clc100bindex:
                                tuples = list(zip(clc100tindex, clc100bindex))
                                multi_index = pd.MultiIndex.from_tuples(tuples, names=["Type", "Column"])
                            else:
                                print(f"Warning: Cannot create multi-index for clc100 ({country}/{newall}). Skipping.")
                                continue
                        elif clcd == "clc10":
                            # Check if required columns exist for index creation
                            if len(clc10tindex) == len(clc10bindex) and clc10bindex:
                                tuples = list(zip(clc10tindex, clc10bindex))
                                multi_index = pd.MultiIndex.from_tuples(tuples, names=["Type", "Column"])
                            else:
                                print(f"Warning: Cannot create multi-index for clc10 ({country}/{newall}). Skipping.")
                                continue
                        else:
                            print(f"Warning: Unknown clcd type '{clcd}'. Skipping multi-index creation.")
                            continue

                        for polytype in poly_layers:
                            try:
                                polyres_base = clcdata[
                                    clcd
                                ].copy()  # Start with a copy of the base CLC data for this type
                                polys = gpd.read_file(file_path, layer=polytype)
                                # Perform spatial analysis
                                polys = points_within_polygons(turbines, polys, turbinecoldict)
                                # Save processed polygons
                                polys.to_file(
                                    gpkgfile, layer=polytype, driver="GPKG", mode="a"
                                )  # Use append mode if file exists

                                # --- Statistical Processing ---
                                # Dynamically find columns based on the current clcd prefix
                                cdcols = [col for col in polys.columns if col.startswith(f"{clcd}_CD_")]
                                valcols = [col for col in polys.columns if col.startswith(f"{clcd}_VAL_")]
                                cdval = cdcols + valcols

                                if not cdval:  # Check if any relevant columns were found/created
                                    print(
                                        f"Warning: No columns starting with '{clcd}_CD_' or '{clcd}_VAL_' found in layer '{polytype}' for {country}/{newall}. Skipping statistics."
                                    )
                                    # Create empty result_df with expected columns to avoid errors later
                                    result_df = pd.DataFrame(columns=["Value", "area", "carea", "cperc", "ashare"])
                                else:
                                    # Calculate statistics
                                    statistics = pd.DataFrame({
                                        "count": polys[cdval].sum().astype(int),
                                    })
                                    # Handle potential empty statistics (if all sums are 0 or columns non-numeric)
                                    if statistics.empty or statistics["count"].isnull().all():
                                        print(
                                            f"Warning: Statistics calculation resulted in empty or all-NaN data for layer '{polytype}' ({country}/{newall})."
                                        )
                                        result_df = pd.DataFrame(columns=["Value", "area", "carea", "cperc", "ashare"])
                                    else:
                                        # Natural sort index (handles numbers correctly)
                                        try:
                                            statistics.index = pd.CategoricalIndex(
                                                statistics.index,
                                                ordered=True,
                                                categories=sorted(
                                                    statistics.index.unique(),  # Use unique values for categories
                                                    key=lambda x: (int("".join(filter(str.isdigit, str(x)))), str(x)),
                                                ),
                                            )
                                            df = statistics.sort_index()
                                        except ValueError as e:
                                            print(
                                                f"Error sorting index for {polytype} ({country}/{newall}): {e}. Using default string sort."
                                            )
                                            df = statistics.sort_index()  # Fallback sort

                                        df.index = df.index.astype(str)  # Convert index to string for extraction
                                        # Extract type and value
                                        # Use regex that handles potential non-matches gracefully with .str[0]
                                        df["type"] = df.index.to_series().str.extract(r"_(CD|VAL)_")[0]
                                        df["value"] = df.index.to_series().str.extract(r"(\d+)$")[0]
                                        # Drop rows where extraction failed (returned NaN)
                                        df = df.dropna(subset=["type", "value"])

                                        if df.empty:
                                            print(
                                                f"Warning: Could not extract type/value from index for '{polytype}' ({country}/{newall})."
                                            )
                                            result_df = pd.DataFrame(
                                                columns=["value", "area", "carea", "cperc", "ashare"]
                                            )
                                        else:
                                            df["value"] = df["value"].astype(int)

                                            # Aggregate CD and VAL counts grouped by Value
                                            total_val_count = df.loc[df["type"] == "VAL", "count"].sum()

                                            # Ensure aggregation handles cases where 'VAL' or 'CD' might be missing for a group
                                            agg_funcs = {
                                                "VAL": pd.NamedAgg(
                                                    column="count",
                                                    aggfunc=lambda x: x[df.loc[x.index, "type"] == "VAL"].sum(),
                                                ),
                                                "CD": pd.NamedAgg(
                                                    column="count",
                                                    aggfunc=lambda x: x[df.loc[x.index, "type"] == "CD"].sum(),
                                                ),
                                            }

                                            result_df = df.groupby("value").agg(**agg_funcs).reset_index()

                                            # Calculate percentages, handle division by zero
                                            result_df["CDp"] = (
                                                result_df["CD"] / result_df["VAL"].replace(0, pd.NA) * 100
                                            ).round(2)
                                            result_df["VALtotp"] = (
                                                result_df["VAL"] / total_val_count if total_val_count else 0
                                            ).round(2)

                                            result_df = result_df.rename(
                                                columns={
                                                    "value": "Value",
                                                    "VAL": "area",
                                                    "CD": "carea",
                                                    "CDp": "cperc",
                                                    "VALtotp": "ashare",
                                                }
                                            )
                                            # Select only the necessary columns after rename
                                            result_df = result_df[["Value", "area", "carea", "cperc", "ashare"]]

                                # Merge results back into the base CLC data
                                # Ensure 'Value' column exists and is the correct type in polyres_base
                                if "Value" in polyres_base.columns:
                                    try:
                                        # Attempt to convert 'Value' to numeric, coercing errors to NaN, then drop NaNs
                                        polyres_base["Value"] = pd.to_numeric(polyres_base["Value"], errors="coerce")
                                        polyres_base = polyres_base.dropna(subset=["Value"])
                                        polyres_base["Value"] = polyres_base["Value"].astype(
                                            int
                                        )  # Convert valid ones to int
                                        polyres_merged = pd.merge(
                                            polyres_base, result_df, on="Value", how="left"
                                        ).fillna(0)
                                        # Ensure area/carea are int after merge/fillna
                                        polyres_merged[["area", "carea"]] = polyres_merged[["area", "carea"]].astype(
                                            int
                                        )

                                    except Exception as merge_err:
                                        print(f"Error merging results for {polytype} ({country}/{newall}): {merge_err}")
                                        # Handle error, maybe skip this polytype or create empty columns
                                        polyres_merged = polyres_base.copy()
                                        for col in ["area", "carea", "cperc", "ashare"]:
                                            polyres_merged[col] = 0  # Add empty result columns
                                else:
                                    print(
                                        f"Warning: 'Value' column not found or invalid in base CLC data for {clcd}. Cannot merge results for {polytype}."
                                    )
                                    polyres_merged = polyres_base.copy()
                                    for col in ["area", "carea", "cperc", "ashare"]:
                                        polyres_merged[col] = 0  # Add empty result columns

                                # Concatenate results horizontally
                                if clcres.empty:
                                    clcres = polyres_merged
                                else:
                                    # Ensure we only concat the result columns and they exist
                                    cols_to_concat = ["area", "carea", "cperc", "ashare"]
                                    if all(col in polyres_merged.columns for col in cols_to_concat):
                                        clcres = pd.concat(
                                            [clcres, polyres_merged[cols_to_concat]],
                                            axis=1,
                                        )
                                    else:
                                        print(
                                            f"Warning: Missing one or more result columns in merged data for {polytype}. Skipping concatenation."
                                        )
                                        # Optionally add empty columns to clcres to maintain structure
                                        missing_cols_df = pd.DataFrame(0, index=clcres.index, columns=cols_to_concat)
                                        clcres = pd.concat([clcres, missing_cols_df], axis=1)

                            except Exception as poly_err:
                                print(
                                    f"Error processing polygon layer '{polytype}' for {clcd} in {country}/{newall}: {poly_err}"
                                )
                                # Decide how to handle: skip polytype, add empty columns?
                                if not clcres.empty:
                                    # Add empty columns if processing failed mid-way for this clcd
                                    empty_res_df = pd.DataFrame(
                                        0, index=clcres.index, columns=["area", "carea", "cperc", "ashare"]
                                    )
                                    clcres = pd.concat([clcres, empty_res_df], axis=1)

                        # --- Write final sheet for this CLC type ---
                        if not clcres.empty and multi_index is not None:
                            # Ensure the number of columns matches the multi-index levels
                            if len(clcres.columns) == len(multi_index):
                                clcres.columns = multi_index  # Apply the multi-index
                                clcres.to_excel(
                                    writer,
                                    sheet_name=clcd,  # , index=False
                                )  # Usually don't write the default numerical index
                            else:
                                print(
                                    f"Error: Column count mismatch ({len(clcres.columns)}) vs MultiIndex ({len(multi_index)}) for {clcd} in {country}/{newall}. Skipping sheet write."
                                )
                        elif clcres.empty:
                            print(f"Info: No results generated for CLC type {clcd} in {country}/{newall}. Empty sheet.")
                            # Optionally write an empty df or skip
                            # pd.DataFrame().to_excel(writer, sheet_name=clcd, index=False)
                        elif multi_index is None:
                            print(
                                f"Info: MultiIndex not created for {clcd} in {country}/{newall}. Skipping sheet write."
                            )

            except Exception as file_err:
                print(f"FATAL ERROR processing base files for {country}/{newall}: {file_err}")
                # Decide whether to continue to the next 'newall' or 'country' or stop


# Example of how to run it (assuming you have the necessary files/folders)
if __name__ == "__main__":
    print("Starting processing...")
    main()
    print("Processing finished.")
