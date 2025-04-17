import os

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# EPSG definition
ccrs = "epsg:3035"  # Equal area Europe

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

gpkg_path = "data/04_clc"  # Path to input GeoPackage
out_path = "data/05_stat"
poly_layers = ["bufferpolys", "concavehulls", "convexhulls"]
resultfields = ["area", "carea", "careap", "ashare"]
clc_file = "data/CLC/CLC.xlsx"
datasets = ["new", "all"]


def main():
    # Set up output folders
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "xls"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "gpkg"), exist_ok=True)

    bindex = resultfields * len(poly_layers)
    tindex = [item for item in poly_layers for _ in range(len(resultfields))]

    # Load CLC data from Excel
    clcdata = {}
    print("Loading CLC data...")
    for clcd in clc_datasets:
        try:
            clcxls = pd.read_excel(clc_file, sheet_name=clcd)
            # Ensure 'Category' and 'CatName' exist before applying groupby/ffill
            if "Category" in clcxls.columns and "CatName" in clcxls.columns:
                clcxls["CatName"] = clcxls.groupby("Category")["CatName"].transform(lambda x: x.ffill())
            else:
                print(f"Warning: 'Category' or 'CatName' not found in sheet {clcd}. Skipping ffill.")
            clcdata[clcd] = clcxls
        except Exception as e:
            print(f"Error loading sheet {clcd} from {clc_file}: {e}")
            clcdata[clcd] = pd.DataFrame()  # Use an empty DataFrame if error occurs

    # Define multi-index templates based on the loaded CLC data
    base_cols_100 = (
        ["Value", "Code", "Name", "Category", "CatName"] if "clc100" in clcdata and not clcdata["clc100"].empty else []
    )
    clc100bindex = base_cols_100 + bindex
    clc100tindex = [""] * len(base_cols_100) + tindex

    base_cols_10 = ["Value", "Name", "Category", "CatName"] if "clc10" in clcdata and not clcdata["clc10"].empty else []
    clc10bindex = base_cols_10 + bindex
    clc10tindex = [""] * len(base_cols_10) + tindex

    # Process each country and dataset
    for country in tqdm(countries, desc="Processing Countries"):
        for newall in tqdm(datasets, desc=f"Processing '{country}' Datasets", leave=False):
            file_path = os.path.join(gpkg_path, country, f"{country}_{newall}.gpkg")
            gpkgfile = os.path.join(out_path, "gpkg", f"{country}_{newall}.gpkg")
            xlsfile = os.path.join(out_path, "xls", f"{country}_{newall}.xlsx")

            try:
                # Read and save turbines immediately
                turbines = gpd.read_file(file_path, layer="turbines")
                # Save turbines to output gpkg (first layer to be written)
                turbines.to_file(gpkgfile, layer="turbines", driver="GPKG")

                # --- Step 1: Read all polygon layers once and store in a dictionary ---
                polys_by_layer = {}
                for polytype in poly_layers:
                    print(f"Reading polygon layer: {polytype}")
                    polys = gpd.read_file(file_path, layer=polytype)
                    # Do not save here; we will save after processing all calculations.
                    polys_by_layer[polytype] = polys

                # --- Step 2: Process CLC-specific calculations for Excel output ---
                with pd.ExcelWriter(xlsfile) as writer:
                    for clcd in clc_datasets:
                        if clcd not in clcdata or clcdata[clcd].empty:
                            print(f"Skipping CLC type {clcd} for {country}/{newall} due to loading errors.")
                            continue

                        clcres = pd.DataFrame()
                        multi_index = None
                        # Set up MultiIndex for Excel output based on CLC type
                        if clcd == "clc100":
                            if len(clc100tindex) == len(clc100bindex) and clc100bindex:
                                tuples = list(zip(clc100tindex, clc100bindex))
                                multi_index = pd.MultiIndex.from_tuples(tuples, names=["Type", "Column"])
                            else:
                                print(f"Warning: Cannot create multi-index for clc100 ({country}/{newall}). Skipping.")
                                continue
                        elif clcd == "clc10":
                            if len(clc10tindex) == len(clc10bindex) and clc10bindex:
                                tuples = list(zip(clc10tindex, clc10bindex))
                                multi_index = pd.MultiIndex.from_tuples(tuples, names=["Type", "Column"])
                            else:
                                print(f"Warning: Cannot create multi-index for clc10 ({country}/{newall}). Skipping.")
                                continue
                        else:
                            print(f"Warning: Unknown clc type '{clcd}'. Skipping multi-index creation.")
                            continue

                        # Loop over each polygon layer for the current clc type
                        for polytype in poly_layers:
                            print(f"Processing CLC '{clcd}', polygon layer: {polytype}")
                            try:
                                # Use a temporary copy of the polygon layer for statistics calculation.
                                polys_temp = polys_by_layer[polytype].copy(deep=True)

                                # --- Apply CLC-specific modifications on the temporary polygons ---
                                # Multiply values of all columns starting with f"{clcd}_CD_" or f"{clcd}_VAL_" by 100.
                                scaling_cols = [
                                    col
                                    for col in polys_temp.columns
                                    if col.startswith(f"{clcd}_CD_") or col.startswith(f"{clcd}_VAL_")
                                ]
                                if scaling_cols:
                                    polys_temp[scaling_cols] = polys_temp[scaling_cols] * 100

                                # Add sum columns (exclude any column ending with _sum already)
                                cd_cols = [
                                    col
                                    for col in polys_temp.columns
                                    if col.startswith(f"{clcd}_CD_") and not col.endswith("_sum")
                                ]
                                val_cols = [
                                    col
                                    for col in polys_temp.columns
                                    if col.startswith(f"{clcd}_VAL_") and not col.endswith("_sum")
                                ]
                                if cd_cols:
                                    polys_temp[f"{clcd}_CD_sum"] = polys_temp[cd_cols].sum(axis=1)
                                if val_cols:
                                    polys_temp[f"{clcd}_VAL_sum"] = polys_temp[val_cols].sum(axis=1)

                                # --- Statistical Processing for Excel ---
                                # Select only the columns that are meant for statistics (exclude the new _sum fields)
                                cdcols = [
                                    col
                                    for col in polys_temp.columns
                                    if col.startswith(f"{clcd}_CD_") and not col.endswith("_sum")
                                ]
                                valcols = [
                                    col
                                    for col in polys_temp.columns
                                    if col.startswith(f"{clcd}_VAL_") and not col.endswith("_sum")
                                ]
                                cdval = cdcols + valcols

                                if not cdval:
                                    print(
                                        f"Warning: No relevant columns for {clcd} in layer '{polytype}' for {country}/{newall}. Skipping statistics."
                                    )
                                    result_df = pd.DataFrame(columns=["Value", "area", "carea", "cperc", "ashare"])
                                else:
                                    statistics = pd.DataFrame({"count": polys_temp[cdval].sum().astype(int)})
                                    if statistics.empty or statistics["count"].isnull().all():
                                        print(
                                            f"Warning: Statistics empty or all-NaN for layer '{polytype}' ({country}/{newall})."
                                        )
                                        result_df = pd.DataFrame(columns=["Value", "area", "carea", "cperc", "ashare"])
                                    else:
                                        try:
                                            statistics.index = pd.CategoricalIndex(
                                                statistics.index,
                                                ordered=True,
                                                categories=sorted(
                                                    statistics.index.unique(),
                                                    key=lambda x: (int("".join(filter(str.isdigit, str(x)))), str(x)),
                                                ),
                                            )
                                            df = statistics.sort_index()
                                        except ValueError as e:
                                            print(
                                                f"Error sorting index for {polytype} ({country}/{newall}): {e}. Using default sort."
                                            )
                                            df = statistics.sort_index()

                                        df.index = df.index.astype(str)
                                        df["type"] = df.index.to_series().str.extract(r"_(CD|VAL)_")[0]
                                        df["value"] = df.index.to_series().str.extract(r"(\d+)$")[0]
                                        df = df.dropna(subset=["type", "value"])

                                        if df.empty:
                                            print(
                                                f"Warning: Could not extract type/value from index for '{polytype}' ({country}/{newall})."
                                            )
                                            result_df = pd.DataFrame(
                                                columns=["Value", "area", "carea", "cperc", "ashare"]
                                            )
                                        else:
                                            df["value"] = df["value"].astype(int)
                                            total_val_count = df.loc[df["type"] == "VAL", "count"].sum()

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
                                            result_df = result_df[["Value", "area", "carea", "cperc", "ashare"]]

                                # Merge computed statistics into the base CLC data
                                polyres_base = clcdata[clcd].copy()
                                if "Value" in polyres_base.columns:
                                    try:
                                        polyres_base["Value"] = pd.to_numeric(polyres_base["Value"], errors="coerce")
                                        polyres_base = polyres_base.dropna(subset=["Value"])
                                        polyres_base["Value"] = polyres_base["Value"].astype(int)
                                        polyres_merged = pd.merge(
                                            polyres_base, result_df, on="Value", how="left"
                                        ).fillna(0)
                                        polyres_merged[["area", "carea"]] = polyres_merged[["area", "carea"]].astype(
                                            int
                                        )
                                    except Exception as merge_err:
                                        print(f"Error merging results for {polytype} ({country}/{newall}): {merge_err}")
                                        polyres_merged = polyres_base.copy()
                                        for col in ["area", "carea", "cperc", "ashare"]:
                                            polyres_merged[col] = 0
                                else:
                                    print(f"Warning: 'Value' column missing in base CLC data for {clcd}.")
                                    polyres_merged = polyres_base.copy()
                                    for col in ["area", "carea", "cperc", "ashare"]:
                                        polyres_merged[col] = 0

                                # Accumulate results (concatenating horizontally)
                                if clcres.empty:
                                    clcres = polyres_merged
                                else:
                                    cols_to_concat = ["area", "carea", "cperc", "ashare"]
                                    if all(col in polyres_merged.columns for col in cols_to_concat):
                                        clcres = pd.concat([clcres, polyres_merged[cols_to_concat]], axis=1)
                                    else:
                                        print(
                                            f"Warning: Some result columns missing for {polytype}. Skipping concatenation."
                                        )
                                        missing_cols_df = pd.DataFrame(0, index=clcres.index, columns=cols_to_concat)
                                        clcres = pd.concat([clcres, missing_cols_df], axis=1)

                            except Exception as poly_err:
                                print(
                                    f"Error processing polygon layer '{polytype}' for {clcd} in {country}/{newall}: {poly_err}"
                                )
                                if not clcres.empty:
                                    empty_res_df = pd.DataFrame(
                                        0, index=clcres.index, columns=["area", "carea", "cperc", "ashare"]
                                    )
                                    clcres = pd.concat([clcres, empty_res_df], axis=1)

                        # Write the accumulated results for the current CLC type to the Excel file.
                        if not clcres.empty and multi_index is not None:
                            if len(clcres.columns) == len(multi_index):
                                clcres.columns = multi_index
                                clcres.to_excel(writer, sheet_name=clcd)
                            else:
                                print(
                                    f"Error: Column count mismatch for {clcd} in {country}/{newall}. Skipping sheet write."
                                )
                        elif clcres.empty:
                            print(f"Info: No results for CLC type {clcd} in {country}/{newall}.")
                        elif multi_index is None:
                            print(f"Info: MultiIndex not created for {clcd} in {country}/{newall}.")
                # --- End of Excel processing ---

                # --- Step 3: Save the modified polygon layers after all calculations ---
                # Here we update each polygon layer in place for all clc types
                # and then write each layer only once to the gpkg file.
                # Remove any existing poly layers in the output GeoPackage if needed.
                if os.path.exists(gpkgfile):
                    os.remove(gpkgfile)
                    # Save turbines again to ensure it is in the output file.
                    turbines.to_file(gpkgfile, layer="turbines", driver="GPKG")

                for i, polytype in enumerate(poly_layers):
                    # Start from the original polygon layer
                    final_polys = polys_by_layer[polytype].copy(deep=True)
                    for clcd in clc_datasets:
                        # Multiply CLC-specific columns (if they exist) by 100.
                        scaling_cols = [
                            col
                            for col in final_polys.columns
                            if col.startswith(f"{clcd}_CD_") or col.startswith(f"{clcd}_VAL_")
                        ]
                        if scaling_cols:
                            final_polys[scaling_cols] = final_polys[scaling_cols] * 100
                        # Add new sum columns.
                        cd_cols = [
                            col
                            for col in final_polys.columns
                            if col.startswith(f"{clcd}_CD_") and not col.endswith("_sum")
                        ]
                        val_cols = [
                            col
                            for col in final_polys.columns
                            if col.startswith(f"{clcd}_VAL_") and not col.endswith("_sum")
                        ]
                        if cd_cols:
                            final_polys[f"{clcd}_CD_sum"] = final_polys[cd_cols].sum(axis=1)
                        if val_cols:
                            final_polys[f"{clcd}_VAL_sum"] = final_polys[val_cols].sum(axis=1)
                    # Write the finalized polygon layer to the GeoPackage.
                    mode = "w" if i == 0 else "a"
                    final_polys.to_file(gpkgfile, layer=polytype, driver="GPKG", mode=mode)

            except Exception as file_err:
                print(f"FATAL ERROR processing base files for {country}/{newall}: {file_err}")


if __name__ == "__main__":
    print("Starting processing...")
    main()
    print("Processing finished.")
