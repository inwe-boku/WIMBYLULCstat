#!/usr/bin/env python3
"""Process turbine point data to create buffer, convex and concave hulls per cluster.

This script:
- Reads GeoPackage point datasets per country/dataset, excluding the extra folder.
- Performs DBSCAN clustering to group turbines.
- Calculates buffer sizes per cluster.
- Generates buffer polygons, convex hulls, and concave hulls.
- Processes all files in input_dir/<extra_folder> as extra datasets.
- Writes outputs under output_dir, with extra outputs in output_dir/<extra_folder>.

Configurable via command-line arguments using tyro.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

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
    """Directory with country subfolders and the extra folder."""

    output_dir: Path = Path("data/02_outhulls/")
    """Directory for processed hull outputs."""

    extra_folder: str = "0_extras"
    """Name of the subfolder for extra datasets (case-sensitive)."""

    country: Optional[str] = None
    """Single country to process; if None, all except extra_folder."""

    datasets: tuple[str, ...] = ("new", "all", "old")
    """Dataset names within each country folder."""

    crs: str = "epsg:3035"
    """CRS for processing."""

    epsilon: float = 1500.0
    """DBSCAN eps parameter."""

    min_samples: int = 2
    """DBSCAN min_samples parameter."""

    buffer_perc: float = 0.5
    """Percentage for buffer calculation."""

    buffer_scale1: float = 2.0 / 3.0
    buffer_scale2: float = 1.0 / 2.0
    """Buffer scale factors."""

    sample_points: int = 99
    concave_ratio: float = 0.25
    """Parameters for concave hull sampling and ratio."""


def get_available_countries(input_dir: Path, extra_folder: str) -> List[str]:
    """List country folders excluding the extra folder."""
    if not input_dir.exists():
        print(f"[WARNING] Input directory '{input_dir}' does not exist.")
        return []
    return sorted(p.name for p in input_dir.iterdir() if p.is_dir() and p.name != extra_folder)


def clustercalculation(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    coords = [(pt.x, pt.y) for pt in gdf.geometry]
    gdf["cluster"] = DBSCAN(eps=cfg.epsilon, min_samples=cfg.min_samples).fit_predict(coords)

    distances = {}
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
    if hull_type == "buffer":
        out = gdf.copy(deep=False)
        out["geometry"] = gdf.buffer(gdf["buffer"] * (cfg.buffer_scale1 + cfg.buffer_scale2))
    elif hull_type == "convex":
        grp = gdf.dissolve(by="cluster")
        hull = grp.convex_hull.buffer(grp["buffer"] * (cfg.buffer_scale1 + cfg.buffer_scale2))
        out = grp.copy(deep=False)
        out["geometry"] = hull
    else:
        bp = sample_buffer_boundaries(gdf, cfg.buffer_scale1, cfg.sample_points)
        grp = bp.dissolve(by="cluster")
        hull = grp.concave_hull(ratio=cfg.concave_ratio).buffer(grp["buffer"] * cfg.buffer_scale2)
        out = grp.copy(deep=False)
        out["geometry"] = hull
    out = out.reset_index(drop=True)
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
    if cfg.country:
        countries = [cfg.country]
    else:
        countries = get_available_countries(cfg.input_dir, cfg.extra_folder)
    if not countries:
        print("[ERROR] No countries found, exiting.")
        return
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
    process_extra_files(cfg)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
