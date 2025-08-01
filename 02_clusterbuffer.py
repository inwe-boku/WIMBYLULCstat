#!/usr/bin/env python3
"""Process turbine point data to create buffer, convex and concave hulls per cluster, with turbine counts.

This script:
- Reads GeoPackage point datasets only for the country specified via --country (none by default).
- Optionally (via --process-extras) processes all files in input_dir/<extra_folder> as extra datasets.
- Performs DBSCAN clustering to group turbines.
- Calculates buffer sizes per cluster.
- Generates buffer polygons, convex hulls, and concave hulls.
- Adds `turbineNo` column with count of turbines per cluster or per feature.
- Writes outputs under output_dir, with extra outputs in output_dir/<extra_folder> if enabled.

Configurable via command-line arguments using tyro.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import tyro
from scipy.spatial import distance
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from tqdm import tqdm


@dataclass
class Config:
    """Configuration for turbine hull processing."""

    input_dir: Path = Path("data/01_outregions/")
    output_dir: Path = Path("data/02_outhulls/")
    extra_folder: str = "0_extras"
    process_extras: bool = False  # If True, also process files in the extras folder
    country: Optional[str] = None  # Only process this country; if None, no country data is processed
    datasets: tuple[str, ...] = ("new", "all", "old")
    crs: str = "epsg:3035"
    epsilon: float = 1500.0
    min_samples: int = 2
    buffer_perc: float = 0.5
    buffer_scale1: float = 2.0 / 3.0
    buffer_scale2: float = 1.0 / 2.0
    sample_points: int = 99
    concave_ratio: float = 0.25


def clustercalculation(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    coords = [(pt.x, pt.y) for pt in gdf.geometry]
    gdf["cluster"] = DBSCAN(eps=cfg.epsilon, min_samples=cfg.min_samples).fit_predict(coords)

    distances: dict[int, float] = {}
    for c in np.unique(gdf["cluster"]):
        if c == -1:
            continue
        pts = gdf.loc[gdf["cluster"] == c, "geometry"]
        arr = np.array([(p.x, p.y) for p in pts])
        dmat = distance.cdist(arr, arr)
        dmat[dmat < 9] = np.inf
        mins = np.min(dmat, axis=1)
        vals = mins[np.isfinite(mins)]
        distances[c] = int(round(np.mean(vals * cfg.buffer_perc), 0)) if len(vals) else np.nan

    gdf["buffer"] = gdf["cluster"].map(distances)
    default = int(round(gdf["buffer"].dropna().mean(), 0)) if not gdf["buffer"].dropna().empty else 0
    gdf["buffer"] = gdf["buffer"].fillna(default)

    gdf["turbineNo"] = gdf.groupby("cluster")["cluster"].transform("count")
    return gdf


def sample_buffer_boundaries(gdf: gpd.GeoDataFrame, scale: float, n: int) -> gpd.GeoDataFrame:
    records = []
    for _, row in gdf.iterrows():
        buf = row.geometry.buffer(row["buffer"] * scale)
        coords = list(buf.boundary.coords)
        step = max(1, len(coords) // n)
        for pt in coords[::step]:
            rec = row.to_dict()
            rec["geometry"] = Point(pt)
            records.append(rec)
    return gpd.GeoDataFrame(records, crs=gdf.crs)


def create_hulls(
    gdf: gpd.GeoDataFrame, hull_type: Literal["buffer", "convex", "concave"], cfg: Config
) -> gpd.GeoDataFrame:
    cluster_counts = gdf.groupby("cluster").size().rename("turbineNo")

    if hull_type == "buffer":
        out = gdf.copy(deep=False)
        out["geometry"] = gdf.buffer(gdf["buffer"] * (cfg.buffer_scale1 + cfg.buffer_scale2))
        out["turbineNo"] = gdf["turbineNo"]

    elif hull_type == "convex":
        grp = gdf.dissolve(by="cluster")
        hull = grp.convex_hull.buffer(grp["buffer"] * (cfg.buffer_scale1 + cfg.buffer_scale2))
        out = grp.copy(deep=False)
        out["geometry"] = hull
        if "turbineNo" in out.columns:
            out = out.drop(columns=["turbineNo"])
        out = out.join(cluster_counts)

    else:
        bp = sample_buffer_boundaries(gdf, cfg.buffer_scale1, cfg.sample_points)
        grp = bp.dissolve(by="cluster")
        hull = grp.concave_hull(ratio=cfg.concave_ratio).buffer(grp["buffer"] * cfg.buffer_scale2)
        out = grp.copy(deep=False)
        out["geometry"] = hull
        if "turbineNo" in out.columns:
            out = out.drop(columns=["turbineNo"])
        out = out.join(cluster_counts)

    out = out.reset_index()
    out["hull_id"] = out.index + 1
    return out


def process_extra_files(cfg: Config) -> None:
    extras = cfg.input_dir / cfg.extra_folder
    if not extras.exists():
        print(f"[WARNING] No extras folder at {extras}")
        return
    fps = [p for p in extras.iterdir() if p.suffix.lower() == ".gpkg"]
    if not fps:
        print(f"[INFO] No GPKG files in {extras}")
        return
    out_ex = cfg.output_dir / cfg.extra_folder
    out_ex.mkdir(parents=True, exist_ok=True)
    print(f"=== Processing extras in {extras} ===")
    for fp in fps:
        print(f"-- {fp.name}")
        gdf = gpd.read_file(fp)
        if gdf.empty:
            print("[INFO] empty, skip")
            continue
        gdf = gdf.to_crs(cfg.crs).drop_duplicates(subset=["geometry"])
        tur = clustercalculation(gdf, cfg)
        stem = fp.stem
        outfp = out_ex / f"{stem}.gpkg"
        tur.to_file(outfp, layer="turbines")
        create_hulls(tur, "buffer", cfg).to_file(outfp, layer="bufferpolys")
        create_hulls(tur[tur["cluster"] != -1], "concave", cfg).to_file(outfp, layer="concavehulls")
        create_hulls(tur[tur["cluster"] != -1], "convex", cfg).to_file(outfp, layer="convexhulls")
        print("[DONE]", outfp)


def main(cfg: Config) -> None:
    # Only process country data if --country is provided
    countries = [cfg.country] if cfg.country else []
    if not countries:
        print("[INFO] No countries specified, skipping country datasets.")
    else:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        for country in tqdm(countries, desc="Countries"):
            od = cfg.output_dir / country
            od.mkdir(exist_ok=True)
            for ds in cfg.datasets:
                fp = cfg.input_dir / country / f"{country}_{ds}.gpkg"
                if not fp.exists():
                    print(f"[WARN] {fp} missing")
                    continue
                gdf = gpd.read_file(fp)
                if gdf.empty:
                    continue
                gdf = gdf.to_crs(cfg.crs).drop_duplicates(subset=["geometry"])
                tur = clustercalculation(gdf, cfg)
                tur.to_file(od / fp.name, layer="turbines")
                create_hulls(tur, "buffer", cfg).to_file(od / fp.name, layer="bufferpolys")
                create_hulls(tur[tur["cluster"] != -1], "concave", cfg).to_file(od / fp.name, layer="concavehulls")
                create_hulls(tur[tur["cluster"] != -1], "convex", cfg).to_file(od / fp.name, layer="convexhulls")

    # Optionally process extra files
    if cfg.process_extras:
        process_extra_files(cfg)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
