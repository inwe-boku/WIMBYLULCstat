#!/usr/bin/env python3
"""
Advanced Raster Sampling for Wind Turbine Geospatial Analysis

This script processes hull and turbine layers to sample raster attributes per country and dataset,
with optional processing of additional GeoPackages in a configurable extras folder.

Key Features:
- Multi-country processing with automatic detection or manual selection, excluding the extras folder by default.
- Multiple raster field sampling (altitude, slope, TRI) with precision control.
- Point geometry: Direct pixel lookup via rasterio.sample for speed.
- Polygon geometry: Mean over pixels via rasterstats.zonal_stats for vectorized performance.
- Optional processing of all .gpkg files in input_dir/<extra_folder> when --process-extras is set.
- CRS reprojection, progress bars, robust error handling.

Usage Examples:
    python process_rasters.py
    python process_rasters.py --countries austria --process-extras
    python process_rasters.py --extra-folder 0_extras --process-extras

Directory Structure Expected:
data/
├── 02_outhulls/       # Input GPKG by country
├── 03_outraster/      # Output directory (created automatically)
└── RASTER/            # Raster files
    ├── Copernicus_DSM_90m_COG_3035.tif
    └── ...

Extras:
Place standalone .gpkg files in input_dir/<extra_folder> (default: "0_extras").
Use --process-extras to include them in sampling; outputs go to output_dir/<extra_folder>.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import rasterio
import tyro
from rasterio.io import DatasetReader
from rasterstats import zonal_stats
from tqdm import tqdm


@dataclass
class Config:
    """Configuration for raster processing operations, including extras folder."""

    input_dir: Path = Path("data/02_outhulls")
    output_dir: Path = Path("data/03_outraster")
    raster_dir: Path = Path("data/RASTER")
    extra_folder: str = "0_extras"


@dataclass
class Args:
    countries: Optional[str] = None
    debug: bool = False
    verbose: bool = False
    dry_run: bool = False
    extra_folder: Optional[str] = None
    process_extras: bool = False
    """When set, process .gpkg files in the extras folder."""


def get_raster_config():
    from dataclasses import dataclass

    @dataclass
    class Spec:
        file: str
        precision: int

    return {
        "altitude": Spec("Copernicus_DSM_90m_COG_3035.tif", 0),
        "slope": Spec("Copernicus_SLOPE_90m_COG_3035.tif", 1),
        "tri": Spec("Copernicus_TRI_90m_COG_3035.tif", 2),
    }


def display_welcome_message():
    print("=" * 80)
    print("🌍 WIND TURBINE RASTER SAMPLING PROCESSOR")
    print("=" * 80)
    print(__doc__)


def get_countries(input_dir: Path, extras: str, override: Optional[List[str]] = None) -> List[str]:
    """Fetch country folder names, excluding the extras folder."""
    if override:
        return [c.strip() for c in override]
    try:
        entries = [p.name for p in input_dir.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return []
    return sorted([e for e in entries if e != extras])


def sample_points(gdf: gpd.GeoDataFrame, rast: DatasetReader, spec, args: Args) -> gpd.GeoDataFrame:
    """Fast point sampling using rasterio.sample."""
    coords = [(pt.x, pt.y) for pt in gdf.geometry]
    samples = list(rast.sample(coords))
    values = []
    for v in samples:
        if v is None or np.isnan(v[0]):
            values.append(np.nan)
        else:
            val = float(v[0])
            values.append(int(val) if spec.precision == 0 else round(val, spec.precision))
    gdf[spec.file] = values
    return gdf


def sample_polygons(gdf: gpd.GeoDataFrame, raster_path: Path, spec, args: Args) -> gpd.GeoDataFrame:
    """Vectorized polygon sampling via rasterstats.zonal_stats."""
    stats = zonal_stats(
        gdf.geometry,
        str(raster_path),
        stats=["mean"],
        nodata=np.nan,
        all_touched=False,
        ignore_nodata=True,
    )
    means = [s.get("mean", np.nan) for s in stats]
    values = []
    for v in means:
        if v is None or np.isnan(v):
            values.append(np.nan)
        else:
            values.append(int(v) if spec.precision == 0 else round(v, spec.precision))
    gdf[spec.file] = values
    return gdf


def sample_layers(layers: dict, rast: DatasetReader, raster_path: Path, spec, args: Args) -> dict:
    """Dispatch sampling per geometry type."""
    out = {}
    for lyr, gdf in layers.items():
        if gdf.empty:
            out[lyr] = gdf
            continue
        if gdf.crs != rast.crs:
            gdf = gdf.to_crs(rast.crs)
        geom_type = gdf.geometry.geom_type.iloc[0].lower()
        if geom_type == "point":
            out[lyr] = sample_points(gdf.copy(), rast, spec, args)
        elif geom_type in ("polygon", "multipolygon"):
            out[lyr] = sample_polygons(gdf.copy(), raster_path, spec, args)
        else:
            if args.debug:
                print(f"Unsupported geometry {geom_type} in {lyr}")
            out[lyr] = gdf
    return out


def process_raster_files(config: Config, args: Args):
    specs = get_raster_config()
    countries = get_countries(
        config.input_dir,
        args.extra_folder or config.extra_folder,
        args.countries.split(",") if args.countries else None,
    )
    for country in tqdm(countries, desc="Countries"):
        in_dir = config.input_dir / country
        out_dir = config.output_dir / country
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for ds in ["new", "all", "old"]:
            gpkg = in_dir / f"{country}_{ds}.gpkg"
            if not gpkg.exists():
                if args.debug:
                    print(f"Missing: {gpkg}")
                continue
            layers = {}
            for lyr in ["turbines", "bufferpolys", "concavehulls", "convexhulls"]:
                try:
                    layers[lyr] = gpd.read_file(gpkg, layer=lyr)
                except Exception as e:
                    if args.debug:
                        print(f"Layer {lyr} error: {e}")
                    layers[lyr] = gpd.GeoDataFrame()
            bbs = [g.total_bounds for g in layers.values() if not g.empty]
            if not bbs:
                continue
            bbox = [
                min(bb[0] for bb in bbs),
                min(bb[1] for bb in bbs),
                max(bb[2] for bb in bbs),
                max(bb[3] for bb in bbs),
            ]
            for name, spec in specs.items():
                raster_path = config.raster_dir / spec.file
                if not raster_path.exists():
                    if args.debug:
                        print(f"Raster missing: {raster_path}")
                    continue
                with rasterio.open(raster_path) as rast:
                    sampled = sample_layers(layers, rast, raster_path, spec, args)
                    layers = sampled
            if not args.dry_run:
                for lyr, gdf in layers.items():
                    out_fp = out_dir / f"{country}_{ds}.gpkg"
                    gdf.to_file(out_fp, layer=lyr, driver="GPKG")
                    if args.debug:
                        print(f"Wrote {lyr} to {out_fp}")

    # process extras if requested
    if args.process_extras:
        extras_dir = config.input_dir / (args.extra_folder or config.extra_folder)
        if extras_dir.exists():
            out_ex = config.output_dir / (args.extra_folder or config.extra_folder)
            if not args.dry_run:
                out_ex.mkdir(parents=True, exist_ok=True)
            fps = [p for p in extras_dir.iterdir() if p.suffix.lower() == ".gpkg"]
            for fp in tqdm(fps, desc="Extras"):
                layers = {}
                for lyr in ["turbines", "bufferpolys", "concavehulls", "convexhulls"]:
                    try:
                        layers[lyr] = gpd.read_file(fp, layer=lyr)
                    except:
                        layers[lyr] = gpd.GeoDataFrame()
                bbs = [g.total_bounds for g in layers.values() if not g.empty]
                if not bbs:
                    continue
                bbox = [
                    min(bb[0] for bb in bbs),
                    min(bb[1] for bb in bbs),
                    max(bb[2] for bb in bbs),
                    max(bb[3] for bb in bbs),
                ]
                for name, spec in specs.items():
                    raster_path = config.raster_dir / spec.file
                    if not raster_path.exists():
                        continue
                    with rasterio.open(raster_path) as rast:
                        sampled = sample_layers(layers, rast, raster_path, spec, args)
                        layers = sampled
                if not args.dry_run:
                    for lyr, gdf in layers.items():
                        out_fp = out_ex / f"{fp.stem}.gpkg"
                        gdf.to_file(out_fp, layer=lyr, driver="GPKG")


def main():
    args = tyro.cli(Args)
    display_welcome_message()
    print("🎯 PROCESSING SUMMARY:")
    extras = args.extra_folder or Config().extra_folder
    print(f"  Extras folder: {extras} (process_extras={args.process_extras})")
    process_raster_files(Config(), args)


if __name__ == "__main__":
    main()
