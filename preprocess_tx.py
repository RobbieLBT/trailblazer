"""
preprocess_tx.py — Transmission line pre-processor for Trailblazer

Runs once (or when source data changes) before build_graph.py.
Writes a cleaned GeoPackage cache that load_transmission() picks up
automatically, skipping the raw shapefile.

Two operations
──────────────
1. Artifact filter
   Walk every LineString's coordinate sequence.  Any consecutive hop that
   exceeds MAX_HOP_NM (default 250 nm) is treated as a bad jump — the
   vertex is dropped and the polyline is split into two independent segments
   at that point.  Both halves are kept.  A line reduced to a single point
   is discarded.

2. Intersection topology
   Find every pair of (cleaned) LineString edges that geometrically
   cross.  At each crossing point, insert a new passable waypoint node and
   split both participating edges at that point.  This gives the router
   real transfer nodes where transmission corridors cross, rather than
   phantom overpasses.

   Intersection nodes are identified by UUID4, stored in a separate
   GeoPackage layer ("intersections") alongside the split edges layer
   ("lines").

Output
──────
   data/eia/tx_preprocessed.gpkg   (default — overridable via --out)

   Layer "lines"         : cleaned, split LineString features
                           retains all original attribute columns
   Layer "intersections" : Point features, columns = [node_id, geometry]

load_transmission() in tx_parser.py checks for this file and, if present,
reads it instead of the raw shapefile.  Delete it to force a rebuild.

Usage
─────
   python preprocess_tx.py [--eia data/eia] [--out data/eia/tx_preprocessed.gpkg]
                           [--bbox min_lon min_lat max_lon max_lat]
                           [--max-hop-nm 250] [--voltage-min 115]
                           [--no-intersections]  # skip step 2 (fast check)
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MAX_HOP_NM:   float = 250.0
DEFAULT_VOLTAGE_MIN:  float = 115.0
DEFAULT_BBOX = (-83.5, 36.0, -75.5, 39.5)   # ORF→CRW AO with 1° pad
DEFAULT_OUT   = "data/eia/tx_preprocessed.gpkg"


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_nm = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R_nm * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _filter_hops(
    coords: list[tuple[float, float]],
    max_hop_nm: float,
) -> list[list[tuple[float, float]]]:
    """
    Walk a (lon, lat) coordinate sequence.  Any hop from the last kept
    vertex that exceeds max_hop_nm is treated as an artifact jump:
      - the offending vertex is dropped
      - the polyline is split at that point — vertices before the jump
        become one segment, vertices after become a new segment

    Returns a list of coordinate lists (one entry per surviving segment).
    Each returned list has >= 2 points.
    """
    if len(coords) < 2:
        return []

    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [coords[0]]

    for i in range(1, len(coords)):
        lon1, lat1 = current[-1]
        lon2, lat2 = coords[i]
        hop_nm = _haversine_nm(lat1, lon1, lat2, lon2)

        if hop_nm > max_hop_nm:
            # Close the current segment if it has at least 2 points
            if len(current) >= 2:
                segments.append(current)
            # Start a fresh segment from the post-jump vertex
            current = [coords[i]]
        else:
            current.append(coords[i])

    if len(current) >= 2:
        segments.append(current)

    return segments


# ── Step 1: Artifact filter ────────────────────────────────────────────────────

def filter_artifacts(gdf, max_hop_nm: float, verbose: bool = True):
    """
    Expand MultiLineStrings, apply hop filter, return a new GeoDataFrame
    where every feature is a clean LineString with no hop > max_hop_nm.

    Features are exploded so one input row may produce multiple output rows;
    all original attribute columns are carried forward.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    rows = []
    original_count = len(gdf)
    dropped_hops   = 0
    produced_segs  = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        line_geoms = (
            [geom]           if geom.geom_type == "LineString"       else
            list(geom.geoms) if geom.geom_type == "MultiLineString"  else []
        )

        attrs = row.drop("geometry").to_dict()

        for line in line_geoms:
            raw = list(line.coords)
            clean_segs = _filter_hops(raw, max_hop_nm)

            # Count how many hops were removed (= vertices lost between segments)
            verts_in  = len(raw)
            verts_out = sum(len(s) for s in clean_segs)
            dropped_hops += verts_in - verts_out  # approximate

            for seg_coords in clean_segs:
                new_row = dict(attrs)
                new_row["geometry"] = LineString(seg_coords)
                rows.append(new_row)
                produced_segs += 1

    result = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    if verbose:
        print(f"  [Artifact filter]  {original_count:,} input features")
        print(f"  [Artifact filter]  {produced_segs:,} clean segments produced")
        print(f"  [Artifact filter]  ~{dropped_hops:,} artifact vertices removed")

    return result


# ── Step 2: Intersection topology ─────────────────────────────────────────────

def build_intersections(gdf, verbose: bool = True):
    """
    Find all pairwise crossing points between LineString edges.
    Split every edge at the crossing points that fall on it.
    Return (split_gdf, intersections_gdf).

    Algorithm
    ─────────
    Naive O(n²) pair-checking is too slow for thousands of edges.
    We use a Shapely STRtree for a spatial index:
      - For each edge, query the tree for candidates whose bounding box
        overlaps → O(n log n) average.
      - For each candidate pair, compute the actual intersection.
      - Collect all intersection Points that land on each edge.
      - Split edges using shapely.ops.split (after snapping the point onto
        the line to avoid floating-point misses).

    Intersection nodes get a UUID4 ident stored in a separate layer.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import split, snap

    lines     = list(gdf.geometry)
    n         = len(lines)
    attrs     = gdf.drop(columns="geometry").to_dict("records")

    # Build STRtree for fast bbox queries
    from shapely.strtree import STRtree
    tree = STRtree(lines)

    # edge_xings[i] = list of Point objects that split edge i
    edge_xings: list[list] = [[] for _ in range(n)]
    xing_points: list[tuple[float, float]] = []   # (lon, lat) deduplicated

    if verbose:
        print(f"  [Intersection]  {n:,} edges — building STRtree index …")

    total_xings = 0

    for i, line_i in enumerate(lines):
        candidate_idxs = tree.query(line_i)
        for j in candidate_idxs:
            if j <= i:
                continue   # process each pair once
            line_j = lines[j]

            inter = line_i.intersection(line_j)
            if inter.is_empty:
                continue

            # Collect individual Points from the intersection
            pts: list[Point] = []
            if inter.geom_type == "Point":
                pts = [inter]
            elif inter.geom_type == "MultiPoint":
                pts = list(inter.geoms)
            elif inter.geom_type == "GeometryCollection":
                pts = [g for g in inter.geoms if g.geom_type == "Point"]
            # LineString intersections (collinear overlap) → skip; no new node

            for pt in pts:
                edge_xings[i].append(pt)
                edge_xings[j].append(pt)
                xing_points.append((pt.x, pt.y))
                total_xings += 1

    if verbose:
        print(f"  [Intersection]  {total_xings:,} crossing points found")

    # ── Split edges ────────────────────────────────────────────────────────────
    SNAP_TOL = 1e-8   # degrees — ~1 mm; keeps split() from missing the point

    split_rows = []
    split_count = 0

    for i, line in enumerate(lines):
        xpts = edge_xings[i]
        row_attrs = attrs[i]

        if not xpts:
            # No intersections — keep as-is
            r = dict(row_attrs)
            r["geometry"] = line
            split_rows.append(r)
            continue

        # Snap all crossing points onto the line, then split iteratively.
        # We split one point at a time; after each split the resulting
        # sub-lines are queued for the next iteration.
        segments = [line]
        for pt in xpts:
            new_segs = []
            snapped_pt = snap(pt, segments[0], SNAP_TOL)   # snap to first seg heuristic
            for seg in segments:
                snapped = snap(pt, seg, SNAP_TOL)
                try:
                    result = split(seg, snapped)
                    new_segs.extend(result.geoms)
                except Exception:
                    new_segs.append(seg)   # split failed — keep whole
            segments = new_segs

        for seg in segments:
            if seg.is_empty or seg.length == 0:
                continue
            r = dict(row_attrs)
            r["geometry"] = seg
            split_rows.append(r)
            split_count += 1

    split_gdf = gpd.GeoDataFrame(split_rows, crs="EPSG:4326")

    # ── Intersection node layer ────────────────────────────────────────────────
    # Deduplicate points within ~10 m of each other
    seen: list[tuple[float, float]] = []
    dedup_pts = []
    DEDUP_DEG = 0.0001   # ~11 m

    for lon, lat in xing_points:
        duplicate = any(
            abs(lon - sx) < DEDUP_DEG and abs(lat - sy) < DEDUP_DEG
            for sx, sy in seen
        )
        if not duplicate:
            seen.append((lon, lat))
            dedup_pts.append({
                "node_id":  str(uuid.uuid4()),
                "geometry": Point(lon, lat),
            })

    xing_gdf = gpd.GeoDataFrame(dedup_pts, crs="EPSG:4326")

    if verbose:
        print(f"  [Intersection]  {split_count:,} split segments produced")
        print(f"  [Intersection]  {len(xing_gdf):,} unique intersection nodes")

    return split_gdf, xing_gdf


# ── Shared loader (mirrors tx_parser.py's ingest logic) ───────────────────────

def _load_raw_gdf(eia_dir: Path, bbox: tuple, voltage_min: float):
    """
    Load and minimally filter the raw EIA shapefile — same logic as
    tx_parser.load_transmission() up to the graph-building step.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box as shapely_box
    from pyproj import Transformer

    # Locate shapefile
    candidates = list(eia_dir.glob("*.shp")) + list(eia_dir.glob("*.gpkg"))
    if not candidates:
        for sub in sorted(eia_dir.iterdir()):
            if sub.is_dir():
                inner = list(sub.glob("*.shp")) + list(sub.glob("*.gpkg"))
                if inner:
                    candidates = inner
                    break
    if not candidates:
        raise FileNotFoundError(f"No shapefile found under {eia_dir}")
    shp_path = candidates[0]
    print(f"\n[TX Preprocess] Source: {shp_path.name}")

    # CRS-aware bbox read
    file_crs = gpd.read_file(shp_path, engine="pyogrio", rows=0).crs
    if file_crs is not None and file_crs.to_epsg() != 4326:
        t = Transformer.from_crs("EPSG:4326", file_crs, always_xy=True)
        x0, y0 = t.transform(bbox[0], bbox[1])
        x1, y1 = t.transform(bbox[2], bbox[3])
        read_bbox = (x0, y0, x1, y1)
    else:
        read_bbox = bbox

    gdf = gpd.read_file(shp_path, engine="pyogrio", bbox=read_bbox)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    clip_geom = shapely_box(*bbox)
    gdf = gdf[gdf.geometry.intersects(clip_geom)].copy()
    print(f"  {len(gdf):,} features in AO")

    # Status filter
    status_col = next(
        (c for c in gdf.columns if c.upper() in ("STATUS",)),
        None
    )
    if status_col:
        mask = gdf[status_col].str.upper().str.contains(
            r"IN.?SERVICE|ENERGIZED|OPERATIONAL", na=True
        )
        gdf = gdf[mask].copy()
        print(f"  {len(gdf):,} in-service after status filter")

    # Voltage filter
    volt_col   = next((c for c in gdf.columns if c.upper() == "VOLTAGE"),   None)
    vclass_col = next((c for c in gdf.columns if c.upper() == "VOLT_CLASS"), None)

    _VOLT_CLASS_MAP = {
        "UNDER 100": 69.0, "100-161": 138.0, "220-287": 230.0,
        "345": 345.0, "500": 500.0, "735 AND ABOVE": 765.0, "DC": 500.0,
    }
    kv = pd.Series(np.nan, index=gdf.index, dtype=float)
    if volt_col:
        raw = pd.to_numeric(gdf[volt_col], errors="coerce")
        kv  = kv.where(~(raw > 0), raw)
    if vclass_col:
        missing = kv.isna()
        mapped  = gdf.loc[missing, vclass_col].str.upper().str.strip().map(_VOLT_CLASS_MAP)
        kv = kv.where(~missing, mapped)

    gdf = gdf[kv.fillna(0) >= voltage_min].copy()
    gdf["_kv"] = kv[gdf.index]
    print(f"  {len(gdf):,} lines at ≥{voltage_min:.0f} kV")

    return gdf


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-process EIA transmission lines: artifact filter + intersection topology"
    )
    parser.add_argument("--eia",         default="data/eia",   help="EIA data directory")
    parser.add_argument("--out",         default=DEFAULT_OUT,  help="Output GeoPackage path")
    parser.add_argument("--max-hop-nm",  type=float, default=DEFAULT_MAX_HOP_NM,
                        help="Drop hops longer than this (nm) [default: 250]")
    parser.add_argument("--voltage-min", type=float, default=DEFAULT_VOLTAGE_MIN,
                        help="Minimum voltage kV [default: 115]")
    parser.add_argument("--bbox",        type=float, nargs=4,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        default=list(DEFAULT_BBOX),
                        help="Bounding box [default: ORF→CRW AO]")
    parser.add_argument("--no-intersections", action="store_true",
                        help="Skip intersection topology step (fast; geometry-only fix)")
    args = parser.parse_args()

    eia_dir = Path(args.eia)
    out_path = Path(args.out)
    bbox     = tuple(args.bbox)

    # ── Load raw data ──────────────────────────────────────────────────────────
    gdf = _load_raw_gdf(eia_dir, bbox, args.voltage_min)

    # ── Step 1: Artifact filter ────────────────────────────────────────────────
    print(f"\n[Step 1] Artifact filter  (max hop = {args.max_hop_nm:.0f} nm)")
    gdf_clean = filter_artifacts(gdf, args.max_hop_nm)

    # ── Step 2: Intersection topology ─────────────────────────────────────────
    if args.no_intersections:
        print("\n[Step 2] Intersection topology — SKIPPED (--no-intersections)")
        split_gdf = gdf_clean
        xing_gdf  = None
    else:
        print("\n[Step 2] Intersection topology")
        split_gdf, xing_gdf = build_intersections(gdf_clean)

    # ── Write output ───────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Output] Writing → {out_path}")

    split_gdf.to_file(out_path, layer="lines",         driver="GPKG")
    if xing_gdf is not None and len(xing_gdf):
        xing_gdf.to_file(out_path, layer="intersections", driver="GPKG")
        print(f"  lines layer        : {len(split_gdf):,} features")
        print(f"  intersections layer: {len(xing_gdf):,} features")
    else:
        print(f"  lines layer        : {len(split_gdf):,} features")
        print(f"  intersections layer: (empty — no crossings found)")

    print("\n[Done]  Run build_graph.py as normal — it will use the cached file.")


if __name__ == "__main__":
    main()
