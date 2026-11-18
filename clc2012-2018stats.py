#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window

D = "u2018_{}_v2020_20u1_raster100m/DATA/U2018_{}_V2020_20u1.tif"
DEFAULTS = {
    "clc2018": D.format("clc2018", "CLC2018"),
    "clc2012": D.format("clc2012", "CLC2012"),
    "cha12": D.format("cha1218", "CHA1218_12"),
    "cha18": D.format("cha1218", "CHA1218_18"),
}
DATASETS = list(DEFAULTS)
NV = 45
VALUES = np.arange(1, NV)
CELL = 10.0
CELL_HA = CELL * CELL / 1e4


class Sampler:
    """Counts raster values on the 10 m grid inside a polygon."""

    def __init__(self, srcs, raster_crs):
        self.srcs, self.crs = srcs, raster_crs
        self.inv = ~srcs["clc2018"].transform
        self.shape = srcs["clc2018"].shape
        self._tf = {}

    def tf(self, zone):
        if zone not in self._tf:
            utm = f"EPSG:{32600 + zone}"
            self._tf[zone] = (Transformer.from_crs(self.crs, utm, always_xy=True),
                              Transformer.from_crs(utm, self.crs, always_xy=True))
        return self._tf[zone]

    def cells(self, geom, zone):
        """Raster row/col of every 10 m UTM cell whose centre lies in the polygon."""
        fwd, back = self.tf(zone)
        g = shapely.transform(geom, lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1])))
        x0, y0, x1, y1 = g.bounds
        x0, y1 = np.floor(x0 / CELL) * CELL, np.ceil(y1 / CELL) * CELL
        w, h = int(np.ceil((x1 - x0) / CELL)) + 1, int(np.ceil((y1 - y0) / CELL)) + 1
        mask = geometry_mask([g], out_shape=(h, w), transform=Affine(CELL, 0, x0, 0, -CELL, y1),
                             invert=True, all_touched=False)
        rr, cc = np.nonzero(mask)
        x, y = back.transform(x0 + (cc + 0.5) * CELL, y1 - (rr + 0.5) * CELL)
        col, row = self.inv * (np.asarray(x), np.asarray(y))
        return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)

    def sample(self, name, row, col):
        """Values 0..44 of raster `name` at the given cells (outside raster -> 0)."""
        out = np.zeros(row.size, dtype=np.int64)
        ok = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        if ok.any():
            r0, r1, c0, c1 = row[ok].min(), row[ok].max() + 1, col[ok].min(), col[ok].max() + 1
            a = self.srcs[name].read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.int64)
            a[(a < 1) | (a >= NV)] = 0
            out[ok] = a[row[ok] - r0, col[ok] - c0]
        return out


def old_columns(gdf, prefix):
    cols = {int(c.rsplit("_", 1)[1]): c for c in gdf.columns if re.fullmatch(prefix + r"_\d+", c)}
    if not cols:
        return None
    out = np.zeros((len(gdf), NV), dtype=np.int64)
    for v, c in cols.items():
        if 0 < v < NV:
            out[:, v] = pd.to_numeric(gdf[c], errors="coerce").fillna(0).astype(np.int64)
    return out


def mismatch(new, old):
    d1 = np.abs(new - old).sum() / 2
    d2 = np.abs(2 * new - old).sum() / 2
    if d1 == 0:
        return 0.0, "identical"
    if old.sum() == 0:
        return d1, "old_empty"
    if d2 <= 1 and old.sum() > 1.5 * new.sum():
        return d2, "old_doubled"
    return d1, "diff"


def shares(df, keys, by, vals):
    g = df.groupby(keys + by, as_index=False, sort=False)[vals].sum()
    for v in vals:
        tot = g.groupby(keys)[v].transform("sum")
        g[v.replace("cells", "pct")] = np.where(tot > 0, 100 * g[v] / tot.where(tot > 0, 1), 0.0)
        g[v.replace("cells", "ha")] = g[v] * CELL_HA
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=".", help="project main directory (default: current dir)")
    p.add_argument("--polygons", default="windparks_europe.gpkg")
    p.add_argument("--layer", default=None)
    p.add_argument("--country-col", default="ct_code")
    p.add_argument("--tile-col", default="tile_best", help="Sentinel-2 tile id (UTM zone)")
    p.add_argument("--legend", default="land-use-codes.csv")
    for d in DATASETS:
        p.add_argument(f"--{d}", default=DEFAULTS[d])
    p.add_argument("--crs", default="EPSG:3035",
                   help="CRS assumed for inputs without a (valid) CRS definition")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    base = Path(args.base)
    out = base / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- inputs ---------------------------------------------------------------
    srcs = {d: rasterio.open(base / getattr(args, d)) for d in DATASETS}
    ref = srcs["clc2018"]
    for d, s in srcs.items():
        if s.shape != ref.shape or not s.transform.almost_equals(ref.transform):
            raise SystemExit(f"Raster '{d}' is not on the same grid as clc2018.")
    raster_crs = ref.crs or rasterio.crs.CRS.from_user_input(args.crs)

    gdf = gpd.read_file(base / args.polygons, layer=args.layer)
    cc = args.country_col
    if cc not in gdf.columns:
        raise SystemExit(f"Column '{cc}' not found. Available: {list(gdf.columns)}")
    # windparks_europe.gpkg carries an "undefined geographic" CRS although the
    # coordinates are ETRS89-LAEA metres -> override instead of reprojecting
    if gdf.crs is None or (gdf.crs.is_geographic and np.abs(gdf.total_bounds).max() > 400):
        print(f"Polygon CRS missing/invalid -> assuming {args.crs}")
        gdf = gdf.set_crs(args.crs, allow_override=True)
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise SystemExit("Polygon dataset contains empty geometries.")
    geoms = gdf.geometry.to_crs(raster_crs).values
    n = len(gdf)

    leg = pd.read_csv(base / args.legend).set_index("Value")
    code = leg["Code"].reindex(range(NV)).fillna(999).astype(int)
    name = leg["Name"].reindex(range(NV)).fillna("Unknown / no data")
    cat = leg["CatName"].reindex(range(NV)).fillna("Unknown")
    catno = leg["Category"].reindex(range(NV)).fillna(0).astype(int)

    # preferred UTM zone: stored zone > S2 tile > centroid
    lon = gdf.geometry.to_crs(raster_crs).centroid.to_crs(4326).x.values
    zone_c = (np.floor((lon + 180) / 6) + 1).astype(int)
    zone_p = zone_c.copy()
    if args.tile_col in gdf.columns:
        zt = pd.to_numeric(gdf[args.tile_col].astype(str).str[:2], errors="coerce")
        zt = zt.where(gdf[args.tile_col].astype(str).str.len() == 5)
        zone_p = np.where(zt.notna(), zt.fillna(0), zone_p).astype(int)
    if "utm_zone" in gdf.columns:
        zone_p = np.where(gdf["utm_zone"].notna(), gdf["utm_zone"].fillna(0), zone_p).astype(int)
    old = old_columns(gdf, "clc100_VAL")

    # ---- 1) counts for every polygon -------------------------------------------
    S = Sampler(srcs, raster_crs)
    cnt = {d: np.zeros((n, NV), dtype=np.int64) for d in DATASETS}
    zone = np.zeros(n, dtype=np.int64)
    diff = np.zeros(n)
    status = np.full(n, "no_reference", dtype=object)
    acc = {}                                        # country -> [t_status, t_cha]
    for i in range(n):
        best = None
        cands = [zone_p[i]] + [z for z in (zone_c[i], zone_c[i] - 1, zone_c[i] + 1)
                               if z != zone_p[i]]
        for z in cands:
            row, col = S.cells(geoms[i], z)
            v18 = S.sample("clc2018", row, col)
            c18 = np.bincount(v18, minlength=NV)
            d, st = (0.0, "no_reference") if old is None else mismatch(c18, old[i])
            if best is None or d < best[0]:
                best = (d, st, z, row, col, v18, c18)
            if old is None or st in ("identical", "old_doubled", "old_empty"):
                break
        diff[i], status[i], zone[i], row, col, v18, cnt["clc2018"][i] = best
        v = {"clc2018": v18}
        for d in DATASETS[1:]:
            v[d] = S.sample(d, row, col)
            cnt[d][i] = np.bincount(v[d], minlength=NV)
        a = acc.setdefault(gdf[cc].iloc[i], [np.zeros((NV, NV), np.int64) for _ in range(2)])
        a[0] += np.bincount(v["clc2012"] * NV + v18, minlength=NV * NV).reshape(NV, NV)
        a[1] += np.bincount(v["cha12"] * NV + v["cha18"], minlength=NV * NV).reshape(NV, NV)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"\r{i + 1}/{n} polygons", end="", flush=True)
    print()

    # ---- GeoPackage with the new columns ----------------------------------------
    new = {}
    for d in DATASETS:
        for v in range(NV):
            # index 0: nodata/unknown for the status layers (written as value 48 if it
            # occurs); for the change layers index 0 = unchanged -> not a VAL column
            if v == 0 and (d.startswith("cha") or not cnt[d][:, 0].any()):
                continue
            if cnt[d][:, v].any() or (d == "clc2018" and old is not None and old[:, v].any()):
                new[f"{d}_VAL_{v if v else 48}"] = cnt[d][:, v]
    # aggregates (Category in the legend): <dataset>_CAT_<category number>
    for d in DATASETS:
        for k in sorted(catno[1:].unique()):
            agg = cnt[d][:, 1:][:, (catno[1:] == k).values].sum(axis=1)
            if agg.any():
                new[f"{d}_CAT_{k}"] = agg
    new = pd.DataFrame(new, index=gdf.index)
    new["utm_zone"] = zone
    new["clc_cells"] = cnt["clc2018"].sum(axis=1)
    new["cha1218_cells"] = cnt["cha12"][:, 1:].sum(axis=1)
    new["cha1218_rel_area"] = 100 * new.cha1218_cells / new.clc_cells.where(new.clc_cells > 0)
    if old is not None:
        new["clc2018_check"] = status
        new["clc2018_diff"] = diff
    gdf_out = gpd.GeoDataFrame(
        pd.concat([gdf.drop(columns=[c for c in new.columns if c in gdf.columns]), new], axis=1),
        geometry=gdf.geometry.name, crs=gdf.crs)
    gpkg_out = out / (Path(args.polygons).stem + "_clc.gpkg")
    gdf_out.to_file(gpkg_out, driver="GPKG")
    print(f"GeoPackage written: {gpkg_out}")
    if old is not None:
        print("Existing clc100_VAL_* vs. new clc2018_VAL_*:",
              pd.Series(status).value_counts().to_dict())

    # ---- 2) statistics per country (within the wind parks) -----------------------
    country = gdf[cc].values
    countries = sorted(pd.unique(country), key=str)
    rows = []
    for c in countries + ["EUROPE"]:
        sel = np.ones(n, bool) if c == "EUROPE" else country == c
        tot = {d: cnt[d][sel].sum(axis=0) for d in DATASETS}
        for v in range(NV):
            if any(tot[d][v] for d in DATASETS[:2]) or (v and (tot["cha12"][v] or tot["cha18"][v])):
                rows.append((c, v if v else 48, code[v], name[v], cat[v], tot["clc2012"][v],
                             tot["clc2018"][v], tot["cha12"][v] if v else 0,
                             tot["cha18"][v] if v else 0))
    base_df = pd.DataFrame(rows, columns=["country", "value", "clc_code", "name", "category",
                                          "cells_2012", "cells_2018", "cells_from", "cells_to"])
    cat_order = list(pd.unique(cat[1:])) + ["Unknown"]
    base_df["category"] = pd.Categorical(base_df.category, cat_order)

    tables, sheets = {}, {}
    for level, by in (("", ["category"]), ("_class", ["value", "clc_code", "name", "category"])):
        for ds, vals in (("clc2012", ["cells_2012"]), ("clc2018", ["cells_2018"]),
                         ("cha1218", ["cells_from", "cells_to"])):
            t = shares(base_df, ["country"], by, vals)
            t = t[(t[vals] != 0).any(axis=1)] if level else t
            if ds == "cha1218":
                # from = class before (loss), to = class after (gain); percent of changed area
                t = t.rename(columns={"cells_from": "cells_2012_loss", "ha_from": "ha_2012_loss",
                                      "pct_from": "pct_2012_loss", "cells_to": "cells_2018_gain",
                                      "ha_to": "ha_2018_gain", "pct_to": "pct_2018_gain"})
                t["net_ha"] = t.ha_2018_gain - t.ha_2012_loss
                area = base_df.groupby("country")["cells_2018"].sum() * CELL_HA
                t["net_pct_of_windpark_area"] = 100 * t.net_ha / t.country.map(area)
                front = by + ["cells_2012_loss", "ha_2012_loss", "pct_2012_loss",
                              "cells_2018_gain", "ha_2018_gain", "pct_2018_gain",
                              "net_ha", "net_pct_of_windpark_area"]
            else:
                y = vals[0][-4:]
                t = t.rename(columns={f"cells_{y}": "cells", f"ha_{y}": "ha", f"pct_{y}": "pct"})
                front = by + ["cells", "ha", "pct"]
            t = t[["country"] + front]
            tables[f"{ds}_by_country{level}"] = t
            if not level:       # percent pivots country x category for the workbook
                for col in [c for c in t.columns if c.startswith("pct")]:
                    pv = t.pivot(index="country", columns="category", values=col).fillna(0)
                    pv = pv.loc[:, (pv != 0).any()]
                    pv["sum"] = pv.sum(axis=1)
                    sheets[f"{ds}_{col}"[:31]] = pv.loc[countries + ["EUROPE"]].reset_index()

    # CLC2018 and CLC2012 side by side per country, by class and by aggregate (category)
    base_df["category_no"] = base_df.value.map(catno).fillna(0).astype(int)
    for key, by in (("class", ["value", "clc_code", "name", "category"]),
                    ("category", ["category_no", "category"])):
        t = shares(base_df, ["country"], by, ["cells_2018", "cells_2012"])
        t = t[(t[["cells_2018", "cells_2012"]] != 0).any(axis=1)]
        t["diff_ha"] = t.ha_2018 - t.ha_2012
        t["diff_pct_points"] = t.pct_2018 - t.pct_2012
        tables[f"summary_by_country_{key}"] = t[
            ["country"] + by + ["cells_2018", "ha_2018", "pct_2018",
                                "cells_2012", "ha_2012", "pct_2012", "diff_ha", "diff_pct_points"]]

    # ---- 3) totals per country + Europe, transitions ------------------------------
    old_cd = old_columns(gdf, "clc100_CD")
    sum_rows, trans_rows = [], []
    for c in countries + ["EUROPE"]:
        sel = np.ones(n, bool) if c == "EUROPE" else country == c
        ts = sum(acc[k][0] for k in acc) if c == "EUROPE" else acc[c][0]
        tc = sum(acc[k][1] for k in acc) if c == "EUROPE" else acc[c][1]
        ms, mc = ts[1:, 1:].copy(), tc[1:, 1:].copy()
        np.fill_diagonal(ms, 0)
        np.fill_diagonal(mc, 0)
        area = cnt["clc2018"][sel].sum() * CELL_HA
        row = {"country": c, "n_polygons": int(sel.sum()), "area_ha": area,
               "changed_ha_clcdiff": ms.sum() * CELL_HA,
               "changed_pct_clcdiff": 100 * ms.sum() * CELL_HA / area,
               "changed_ha_cha1218": mc.sum() * CELL_HA,
               "changed_pct_cha1218": 100 * mc.sum() * CELL_HA / area}
        if old_cd is not None and old is not None:      # existing Sentinel-2 change detection
            row["old_area_ha_clc100"] = old[sel].sum() * CELL_HA
            row["old_changed_ha_s2cd"] = old_cd[sel].sum() * CELL_HA
            row["old_changed_pct_s2cd"] = 100 * old_cd[sel].sum() / max(old[sel].sum(), 1)
        sum_rows.append(row)
        for f, t in zip(*np.nonzero(ms + mc)):
            trans_rows.append((c, code[f + 1], code[t + 1], name[f + 1], name[t + 1],
                               cat[f + 1], cat[t + 1], ms[f, t] * CELL_HA, mc[f, t] * CELL_HA))
    tables["summary_by_country"] = summ = pd.DataFrame(sum_rows)
    tables["transitions_by_country"] = pd.DataFrame(
        trans_rows, columns=["country", "clc_from", "clc_to", "name_from", "name_to",
                             "category_from", "category_to", "ha_clcdiff", "ha_cha1218"]
    ).sort_values(["country", "ha_cha1218"], ascending=[True, False])

    if old is not None:
        chk = pd.DataFrame({"polygon": np.arange(n), "country": country, "utm_zone": zone,
                            "old_clc100_VAL_sum": old.sum(axis=1),
                            "new_clc2018_VAL_sum": cnt["clc2018"].sum(axis=1),
                            "check": status, "diff_cells": diff})
        for c in ("NUTS3_CODE", "cluster", args.tile_col):
            if c in gdf.columns:
                chk.insert(2, c, gdf[c].values)
        sheets["check_not_identical"] = chk[chk.check != "identical"]

    order = ["summary_by_country", "summary_by_country_class", "summary_by_country_category",
             "clc2012_by_country", "clc2018_by_country",
             "cha1218_by_country", "clc2012_by_country_class", "clc2018_by_country_class",
             "cha1218_by_country_class", "transitions_by_country"]
    with pd.ExcelWriter(out / "clc_change_stats.xlsx") as xw:
        for k in order:
            tables[k].to_csv(out / f"{k}.csv", index=False)
            tables[k].to_excel(xw, sheet_name=k, index=False)
        for k, df in sheets.items():
            df.to_excel(xw, sheet_name=k, index=False)

    print(summ.round(3).to_string(index=False))
    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
