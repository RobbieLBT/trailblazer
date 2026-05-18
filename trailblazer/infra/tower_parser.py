"""
trailblazer/infra/tower_parser.py — FCC ASR cell tower mesh graph (Phase 3)

Reads FCC Antenna Structure Registration data and builds a Delaunay-
triangulated mesh graph for C2-link-aware BVLOS routing.

Download
────────
1. Go to https://www.fcc.gov/uls/transactions/daily-weekly
2. Under "Antenna Structure Registration" → "Complete ASR DB" → download l_asr.zip
3. Unzip — you need CO.dat (coordinates) and optionally RA.dat (height/owner)
4. Place both at data/fcc/CO.dat and data/fcc/RA.dat

CO.dat column layout (confirmed from field inspection)
──────────────────────────────────────────────────────
Pipe-delimited, no header. Record type "CO" at index 0.

  0   record_type       "CO"
  1   content_ind       "REG" | "APP"
  2   file_number       e.g. "A0000039"
  3   unique_sys_id     numeric, key for RA.dat join
  4   registration_num  e.g. "96973"
  5   struct_code       "T" = tower (FCC type code)
  6   lat_deg
  7   lat_min
  8   lat_sec
  9   lat_dir           "N" | "S"
 10   lat_arcsec_total  redundant; skip
 11   lon_deg
 12   lon_min
 13   lon_sec
 14   lon_dir           "E" | "W"
 15   lon_arcsec_total  redundant; skip
 16+  (empty trailing fields)

Height is NOT in CO.dat — it is in RA.dat if present.
All ASR-registered structures already meet the FAA notification threshold
(≥ 60m AGL or near airports) so bbox filtering is sufficient without height.
If RA.dat is present, height is extracted and stored for C2 margin calculations.

Graph strategy
──────────────
1. Parse CO.dat → {sys_id: (lat, lon)}
2. Optionally parse RA.dat → {sys_id: height_agl_m}
3. Filter to bbox
4. Grid-cell reduction (keep tallest/any tower per GRID_CELL_KM cell)
5. Delaunay triangulation, drop edges > MAX_EDGE_KM
6. C2_margin on each edge: min(h1, h2) / dist_km
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import Fix, AirwaySegment, GraphData


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_EDGE_KM:  float = 60.0    # drop Delaunay edges longer than this
GRID_CELL_KM: float = 25.0    # one tower per N-km cell
DEFAULT_BBOX        = (-83.5, 36.0, -75.5, 39.5)  # ORF→CRW AO


# ── Geometry ──────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _dms(deg: str, mn: str, sec: str, direction: str) -> Optional[float]:
    try:
        d, m, s, dr = (
            str(deg).strip(), str(mn).strip(),
            str(sec).strip(), str(direction).strip()
        )
        # Reject any field that pandas left as NaN or empty
        if any(v.lower() in ("nan", "none", "") for v in (d, m, s, dr)):
            return None
        val = float(d) + float(m) / 60.0 + float(s) / 3600.0
        if dr.upper() in ("S", "W"):
            val = -val
        return val
    except (ValueError, TypeError):
        return None


# ── File parsers ──────────────────────────────────────────────────────────────

def _parse_co(co_path: Path, bbox: tuple) -> dict[str, tuple[float, float]]:
    """
    Parse CO.dat → {unique_sys_id: (lat, lon)}.
    Uses pandas for fast bulk parse (~10× faster than Python loop on 200k rows).
    """
    import pandas as pd

    min_lon, min_lat, max_lon, max_lat = bbox

    # Column layout confirmed from field inspection:
    #   0=record_type  1=content_ind  2=file_num  3=unique_sys_id  4=reg_num
    #   5=struct_code  6=lat_deg  7=lat_min  8=lat_sec  9=lat_dir
    #   10=lat_arcsec(skip)  11=lon_deg  12=lon_min  13=lon_sec  14=lon_dir
    #   15=lon_arcsec(skip)  16+=empty
    try:
        df = pd.read_csv(
            co_path,
            sep="|",
            header=None,
            dtype=str,
            encoding="latin-1",
            on_bad_lines="skip",
            engine="python",
        )
    except Exception as exc:
        print(f"  [Tower] pandas read failed ({exc}), falling back to line parser")
        return _parse_co_fallback(co_path, bbox)

    # Keep only CO records with enough columns
    df = df[df[0].str.strip() == "CO"].copy()
    if len(df) == 0 or df.shape[1] < 15:
        print("  [Tower] No CO records found in file")
        return {}

    def _col_dms(deg_col, min_col, sec_col, dir_col) -> pd.Series:
        # Cast to str so NaN fields become the string "nan", handled by _dms
        sub = df[[deg_col, min_col, sec_col, dir_col]].astype(str)
        def row_dms(row):
            return _dms(row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3])
        return sub.apply(row_dms, axis=1)

    df["_lat"] = _col_dms(6, 7, 8, 9)
    df["_lon"] = _col_dms(11, 12, 13, 14)
    df["_sys"] = df[3].str.strip()

    # Drop invalid
    df = df.dropna(subset=["_lat", "_lon", "_sys"])
    df = df[df["_sys"] != ""]

    # Bbox filter
    df = df[
        (df["_lat"] >= min_lat) & (df["_lat"] <= max_lat) &
        (df["_lon"] >= min_lon) & (df["_lon"] <= max_lon)
    ]

    coords = dict(zip(df["_sys"], zip(df["_lat"], df["_lon"])))
    print(f"  [Tower] CO.dat: {len(coords):,} towers in AO "
          f"(from {len(df):,} valid CO records)")
    return coords


def _parse_co_fallback(co_path: Path, bbox: tuple) -> dict[str, tuple[float, float]]:
    """Line-by-line fallback if pandas fails."""
    min_lon, min_lat, max_lon, max_lat = bbox
    coords: dict[str, tuple[float, float]] = {}
    with open(co_path, encoding="latin-1", errors="replace") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 15 or parts[0].strip() != "CO":
                continue
            lat = _dms(parts[6], parts[7], parts[8], parts[9])
            lon = _dms(parts[11], parts[12], parts[13], parts[14])
            if lat is None or lon is None:
                continue
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            sys_id = parts[3].strip()
            if sys_id:
                coords[sys_id] = (lat, lon)
    return coords


def _parse_ra(ra_path: Path, sys_ids: set[str]) -> dict[str, float]:
    """
    Parse RA.dat → {unique_sys_id: height_agl_m} using pandas for speed.
    Scans non-empty trailing numeric fields for AGL height (30–3000 ft range).
    """
    import pandas as pd

    heights: dict[str, float] = {}
    if not ra_path.exists():
        return heights

    try:
        df = pd.read_csv(
            ra_path,
            sep="|",
            header=None,
            dtype=str,
            encoding="latin-1",
            on_bad_lines="skip",
            engine="python",
        )
    except Exception:
        return heights

    df = df[df[0].str.strip() == "RA"].copy()
    if len(df) == 0:
        return heights

    # Filter to bbox-matched towers only
    df["_sys"] = df[3].str.strip()
    df = df[df["_sys"].isin(sys_ids)]
    if len(df) == 0:
        return heights

    # Find height: scan rightmost numeric columns for value in [30, 3000] ft
    def _find_height(row) -> Optional[float]:
        for v in reversed(row.values):
            s = str(v).strip()
            if not s or s in ("nan", "None"):
                continue
            try:
                h = float(s)
                if 30.0 <= h <= 3000.0:
                    return h * 0.3048
            except ValueError:
                continue
        return None

    df["_h"] = df.apply(_find_height, axis=1)
    df = df.dropna(subset=["_h"])
    heights = dict(zip(df["_sys"], df["_h"]))

    print(f"  [Tower] RA.dat: height found for {len(heights):,} / {len(sys_ids):,} towers")
    return heights


# ── Grid reduction ────────────────────────────────────────────────────────────

def _grid_reduce(
    towers: list[dict],
    cell_km: float,
) -> list[dict]:
    """Keep the tallest (or first) tower per grid cell."""
    # Use mid-AO latitude for lon cell size
    cell_lat = cell_km / 111.0
    cell_lon = cell_km / (111.0 * math.cos(math.radians(38.0)))

    cells: dict[tuple[int, int], dict] = {}
    for t in towers:
        # math.floor(), not int() — int() truncates toward zero, which gives the
        # wrong cell index for negative longitudes (i.e. everywhere in the AO).
        # e.g. int(-564.3) = -564 but floor(-564.3) = -565.  The error is small
        # per tower but accumulates to visible gaps when grid_cell_km is small.
        ci = math.floor(t["lat"] / cell_lat)
        cj = math.floor(t["lon"] / cell_lon)
        key = (ci, cj)
        existing = cells.get(key)
        if existing is None:
            cells[key] = t
        elif t.get("height_m", 0) > existing.get("height_m", 0):
            cells[key] = t

    result = list(cells.values())
    print(f"  [Tower] {len(result):,} towers after {cell_km:.0f} km grid reduction "
          f"(from {len(towers):,})")
    return result


# ── Delaunay mesh ─────────────────────────────────────────────────────────────

def _build_mesh(
    towers: list[dict],
    max_edge_km: float,
) -> tuple[dict[str, Fix], list[AirwaySegment], dict[tuple[str, str], float]]:
    from scipy.spatial import Delaunay

    if len(towers) < 3:
        raise ValueError(
            f"Need at least 3 towers for triangulation, got {len(towers)}."
        )

    coords = np.array([[t["lat"], t["lon"]] for t in towers])
    tri    = Delaunay(coords)

    fixes: dict[str, Fix] = {}
    for t in towers:
        ident = f"T{t['sys_id']}"
        fixes[ident] = Fix(
            ident=ident,
            lat=t["lat"],
            lon=t["lon"],
            fix_type="TOWER",
            name=t.get("owner", "")[:50],
            elevation_m=t.get("height_m"),  # tower height ≠ terrain elevation;
            # stored here as a proxy; overwritten by DEM fetch if --elevation used
        )

    seen: set[frozenset] = set()
    segments: list[AirwaySegment] = []
    c2_map: dict[tuple[str, str], float] = {}
    dropped = 0

    for simplex in tri.simplices:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            a, b = simplex[i], simplex[j]
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)

            ta, tb = towers[a], towers[b]
            dist_km = _haversine_km(ta["lat"], ta["lon"], tb["lat"], tb["lon"])
            if dist_km > max_edge_km:
                dropped += 1
                continue

            ha = ta.get("height_m", 30.0)
            hb = tb.get("height_m", 30.0)
            c2 = min(ha, hb) / max(dist_km, 0.1)

            ia, ib = f"T{ta['sys_id']}", f"T{tb['sys_id']}"
            for from_id, to_id in ((ia, ib), (ib, ia)):
                seg = AirwaySegment(
                    airway_id="TOWER_MESH",
                    from_ident=from_id,
                    to_ident=to_id,
                    seq_from=0,
                    seq_to=1,
                    voltage_kv=0.0,
                )
                segments.append(seg)
                c2_map[(from_id, to_id)] = c2

    print(f"  [Tower] {len(segments)//2:,} mesh edges "
          f"({dropped} dropped > {max_edge_km:.0f} km)")
    return fixes, segments, c2_map


# ── TowerGraphData ─────────────────────────────────────────────────────────────

class TowerGraphData(GraphData):
    """GraphData subclass carrying per-edge C2 margin values."""
    def __init__(self, fixes, segments, c2_margins):
        super().__init__(fixes=fixes, segments=segments, source="cell")
        self.c2_margins: dict[tuple[str, str], float] = c2_margins


# ── Main loader ───────────────────────────────────────────────────────────────

def load_towers(
    fcc_dir:      str | Path = "data/fcc",
    bbox:         tuple | None = None,
    max_edge_km:  float = MAX_EDGE_KM,
    grid_cell_km: float = GRID_CELL_KM,
) -> TowerGraphData:
    """
    Build a cell tower mesh GraphData from FCC ASR CO.dat (+ optional RA.dat).

    Parameters
    ----------
    fcc_dir      : directory containing CO.dat (and optionally RA.dat)
    bbox         : (min_lon, min_lat, max_lon, max_lat)
                   Default: ORF→CRW AO
    max_edge_km  : drop Delaunay edges longer than this (default 60 km)
    grid_cell_km : grid-reduction cell size (default 25 km)
    """
    if bbox is None:
        bbox = DEFAULT_BBOX

    fcc_dir = Path(fcc_dir)
    co_path = fcc_dir / "CO.dat"
    ra_path = fcc_dir / "RA.dat"

    # Try lowercase too
    if not co_path.exists() and (fcc_dir / "co.dat").exists():
        co_path = fcc_dir / "co.dat"
    if not ra_path.exists() and (fcc_dir / "ra.dat").exists():
        ra_path = fcc_dir / "ra.dat"

    if not co_path.exists():
        raise FileNotFoundError(
            f"CO.dat not found at {fcc_dir}/CO.dat\n\n"
            f"Download:\n"
            f"  1. https://www.fcc.gov/uls/transactions/daily-weekly\n"
            f"  2. Antenna Structure Registration → Complete ASR DB → l_asr.zip\n"
            f"  3. Unzip and place CO.dat (and RA.dat) in {fcc_dir}/\n"
        )

    print(f"\n[Tower Parser] fcc_dir: {fcc_dir}")
    print(f"  [Tower] Bbox: lon {bbox[0]}..{bbox[2]}  lat {bbox[1]}..{bbox[3]}")

    # Step 1: coordinates from CO.dat
    coord_map = _parse_co(co_path, bbox)
    if not coord_map:
        raise ValueError(
            "No towers found in AO. Verify CO.dat covers your target region."
        )

    # Step 2: heights from RA.dat (optional)
    height_map = _parse_ra(ra_path, set(coord_map.keys()))

    # Step 3: assemble tower list
    towers = []
    for sys_id, (lat, lon) in coord_map.items():
        towers.append({
            "sys_id":   sys_id,
            "lat":      lat,
            "lon":      lon,
            "height_m": height_map.get(sys_id, 30.0),  # fallback 30m ≈ 100ft
        })

    # Step 4: grid reduction
    towers = _grid_reduce(towers, grid_cell_km)

    # Step 5: Delaunay mesh
    fixes, segments, c2_map = _build_mesh(towers, max_edge_km)

    print(f"\n  [Tower] Graph ready: {len(fixes):,} nodes, "
          f"{len(segments):,} directed segments\n")

    return TowerGraphData(fixes=fixes, segments=segments, c2_margins=c2_map)