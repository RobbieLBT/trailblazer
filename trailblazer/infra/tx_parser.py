"""
trailblazer/infra/tx_parser.py — EIA transmission line graph ingestion

Reads a single HIFLD shapefile and produces a GraphData:

    data/eia/Electric_Power_Transmission_Lines/  → line geometries + voltage

Download
────────
1. Go to https://hifld-geoplatform.hub.arcgis.com
2. Search "Electric Power Transmission Lines"
3. Download → Shapefile → unzip into data/eia/
   Expected layout after unzip:
     data/eia/Electric_Power_Transmission_Lines.shp   (or similar name)
   OR a subdirectory:
     data/eia/Electric_Power_Transmission_Lines/Electric_Power_Transmission_Lines.shp

That is the only EIA file needed.  No substation download required.

Node strategy
─────────────
The transmission lines shapefile has SUB_1 and SUB_2 fields naming the
substation at each end of every line.  Those names become node idents directly,
so lines sharing a substation are naturally connected without any separate
substation layer.  For lines with blank endpoint names, a synthetic ident is
derived from the coordinate hash.

If a pre-processed GeoPackage exists at data/eia/tx_preprocessed.gpkg
(written by preprocess_tx.py), it is used instead of the raw shapefile.
The cache contains artifact-filtered, intersection-split LineStrings and a
separate layer of UUID-ident passable waypoints at every crossing point.
Delete the .gpkg to force a rebuild from raw data.

Long lines (> MAX_SEGMENT_NM) are split at intermediate waypoints so the
weather provider is sampled at fine-enough spatial resolution.  A proximity
registry merges endpoints that are within SNAP_THRESHOLD_KM of each other,
which handles slight misalignments between adjacent line endpoints.

Ident scheme
────────────
    Name-derived   : cleaned SUB_1 / SUB_2 text, e.g. "NORFOLK 500KV"
    Synthetic      : "TX{hash:05d}" for lines with no substation name
    Intersection   : UUID4 string for passable crossing-point waypoints

airway_id encodes the nominal voltage class, e.g. "500KV", "345KV".
voltage_kv on AirwaySegment feeds the social-impact discount table (Phase 4):
    765 kV → 0.10×    500 kV → 0.20×    345 kV → 0.40×
    230 kV → 0.65×    115 kV → 0.85×    0 (off-ROW) → 1.00×
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..types import Fix, AirwaySegment, GraphData

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SEGMENT_NM:    float = 25.0   # split line segments longer than this
MIN_VOLTAGE_KV:    float = 115.0  # exclude sub-transmission
SNAP_THRESHOLD_KM: float = 1.0   # merge endpoints within this distance

_VOLT_CLASS_MAP: dict[str, float] = {
    "UNDER 100":      69.0,
    "100-161":       138.0,
    "220-287":       230.0,
    "345":           345.0,
    "500":           500.0,
    "735 AND ABOVE": 765.0,
    "DC":            500.0,
}


def _voltage_label(kv: float) -> str:
    if kv >= 700: return "765KV"
    if kv >= 450: return "500KV"
    if kv >= 300: return "345KV"
    if kv >= 200: return "230KV"
    if kv >= 100: return "115KV"
    return "TXLOW"


# ── File detection ────────────────────────────────────────────────────────────

def _find_shapefile(eia_dir: Path) -> Path:
    """
    Locate the transmission lines shapefile under eia_dir.
    Accepts: a .shp/.gpkg file directly, a directory containing one,
    or a subdirectory (one level deep) containing one.
    """
    if eia_dir.suffix in (".shp", ".gpkg") and eia_dir.exists():
        return eia_dir

    candidates = list(eia_dir.glob("*.shp")) + list(eia_dir.glob("*.gpkg"))
    if candidates:
        return candidates[0]

    for sub in sorted(eia_dir.iterdir()):
        if sub.is_dir():
            inner = list(sub.glob("*.shp")) + list(sub.glob("*.gpkg"))
            if inner:
                return inner[0]

    raise FileNotFoundError(
        f"No transmission lines shapefile found under {eia_dir}\n\n"
        f"Download instructions:\n"
        f"  1. Go to https://hifld-geoplatform.hub.arcgis.com\n"
        f"  2. Search 'Electric Power Transmission Lines'\n"
        f"  3. Download → Shapefile\n"
        f"  4. Unzip into {eia_dir}\n"
    )


def _get_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _numeric_voltage(gdf, volt_col, vclass_col) -> pd.Series:
    kv = pd.Series(np.nan, index=gdf.index, dtype=float)
    if volt_col:
        raw = pd.to_numeric(gdf[volt_col], errors="coerce")
        kv  = kv.where(~(raw > 0), raw)
    if vclass_col:
        missing = kv.isna()
        mapped  = gdf.loc[missing, vclass_col].str.upper().str.strip().map(_VOLT_CLASS_MAP)
        kv = kv.where(~missing, mapped)
    return kv


# ── Ident helpers ─────────────────────────────────────────────────────────────

def _clean_name(raw) -> str:
    """Normalise a SUB_1/SUB_2 field into a short stable ident string."""
    s = str(raw).strip()
    if s.lower() in ("", "nan", "none", "unknown", "not available"):
        return ""
    s = re.sub(r"[^\w\s\-]", "", s.upper()).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:40]


def _synthetic_ident(lat: float, lon: float) -> str:
    h = abs(hash((round(lat, 4), round(lon, 4)))) % 100_000
    return f"TX{h:05d}"


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _interpolate_line(
    coords: list[tuple[float, float]],
    max_spacing_nm: float,
) -> list[tuple[float, float]]:
    """Resample a (lon, lat) polyline so no consecutive gap exceeds max_spacing_nm."""
    if len(coords) < 2:
        return coords
    result = [coords[0]]
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        seg_nm = _haversine_km(lat1, lon1, lat2, lon2) / 1.852
        n = max(1, int(math.ceil(seg_nm / max_spacing_nm)))
        for j in range(1, n + 1):
            t = j / n
            result.append((lon1 + t * (lon2 - lon1), lat1 + t * (lat2 - lat1)))
    return result


# ── Node registry ─────────────────────────────────────────────────────────────

class _NodeRegistry:
    """
    Deduplicates graph nodes by proximity and by name.

    Uses a grid-based spatial index for O(1) proximity lookups instead of
    an O(n) linear scan.  For a 2,000-node transmission graph this reduces
    ~2M haversine calls to ~18 per lookup (9 grid cells × ~2 neighbours).

    Grid cell size = snap_threshold_km so the 3×3 neighbourhood is sufficient
    to catch all candidates within the snap radius.
    """

    def __init__(self, snap_km: float = SNAP_THRESHOLD_KM) -> None:
        self._snap_km   = snap_km
        self._nodes:    dict[str, Fix]               = {}
        self._names:    dict[str, str]               = {}   # name → ident
        # Grid index: (cell_i, cell_j) → [(lat, lon, ident), ...]
        self._grid:     dict[tuple, list]            = {}
        # Precompute cell sizes in degrees (approximate, good for continental US)
        self._cell_lat  = snap_km / 111.0
        self._cell_lon  = snap_km / (111.0 * math.cos(math.radians(38.0)))

    def _cell(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(lat / self._cell_lat), int(lon / self._cell_lon))

    def _candidates(self, lat: float, lon: float):
        """Yield (lat, lon, ident) tuples from the 3×3 neighbourhood."""
        ci, cj = self._cell(lat, lon)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for entry in self._grid.get((ci + di, cj + dj), []):
                    yield entry

    def get_or_create(
        self,
        lat: float,
        lon: float,
        name: str = "",
        state: str = "",
    ) -> str:
        # 1. Name-based lookup — same substation name always maps to the same node
        if name and name in self._names:
            return self._names[name]

        # 2. Proximity lookup via grid index — O(1) instead of O(n)
        for node_lat, node_lon, node_id in self._candidates(lat, lon):
            if _haversine_km(lat, lon, node_lat, node_lon) <= self._snap_km:
                # Upgrade the node name if it was previously anonymous
                if name and not self._nodes[node_id].name:
                    self._nodes[node_id] = Fix(
                        ident=node_id, lat=node_lat, lon=node_lon,
                        fix_type="TX_NODE", name=name[:50], state=state[:2],
                    )
                    self._names[name] = node_id
                return node_id

        # 3. Create new node
        ident = name if name else _synthetic_ident(lat, lon)
        base, counter = ident, 0
        while ident in self._nodes:
            counter += 1
            ident = f"{base}_{counter}"

        self._nodes[ident] = Fix(
            ident=ident, lat=lat, lon=lon,
            fix_type="TX_NODE", name=name[:50] if name else "", state=state[:2],
        )
        ci, cj = self._cell(lat, lon)
        self._grid.setdefault((ci, cj), []).append((lat, lon, ident))
        if name:
            self._names[name] = ident
        return ident

    @property
    def fixes(self) -> dict[str, Fix]:
        return dict(self._nodes)


# ── Pre-processed cache loader ────────────────────────────────────────────────

def _load_preprocessed(
    gpkg_path: Path,
    max_segment_nm:    float = MAX_SEGMENT_NM,
    snap_threshold_km: float = SNAP_THRESHOLD_KM,
) -> GraphData:
    """
    Build a GraphData from a pre-processed GeoPackage produced by
    preprocess_tx.py.  The 'lines' layer contains cleaned, split LineStrings;
    the optional 'intersections' layer supplies UUID-ident passable waypoints
    at crossing points.

    Processing mirrors load_transmission() from the point where raw GDF rows
    are walked to produce nodes and AirwaySegments — artifact filtering and
    intersection-finding already happened offline.
    """
    import geopandas as gpd

    print(f"\n[TX Parser] Using pre-processed cache: {gpkg_path.name}")

    gdf = gpd.read_file(gpkg_path, layer="lines")
    print(f"  [TX]  {len(gdf):,} pre-processed segments")

    # Pre-register intersection nodes by UUID so both split sub-edges that
    # meet at a crossing point resolve to the same stable ident.
    try:
        xing_gdf = gpd.read_file(gpkg_path, layer="intersections")
        xing_nodes: dict[tuple, str] = {}   # (rounded_lon, rounded_lat) → uuid
        for _, xrow in xing_gdf.iterrows():
            key = (round(xrow.geometry.x, 5), round(xrow.geometry.y, 5))
            xing_nodes[key] = str(xrow["node_id"])
        print(f"  [TX]  {len(xing_nodes):,} intersection nodes pre-registered")
    except Exception:
        xing_nodes = {}

    # Column detection (same candidates as load_transmission)
    volt_col   = _get_col(gdf, ["VOLTAGE",    "voltage",   "Voltage",  "VOLT"])
    vclass_col = _get_col(gdf, ["VOLT_CLASS", "voltclass", "VOLTCLASS"])
    sub1_col   = _get_col(gdf, ["SUB_1",  "sub_1",  "FROM_SUB", "FROMSUB", "from_sub"])
    sub2_col   = _get_col(gdf, ["SUB_2",  "sub_2",  "TO_SUB",   "TOSUB",   "to_sub"])
    state_col  = _get_col(gdf, ["STATE",  "state",  "STATE_NAME"])
    kv_col     = _get_col(gdf, ["_kv"])

    registry = _NodeRegistry(snap_threshold_km)

    # Pre-register intersection nodes into the registry
    for (ilon, ilat), iid in xing_nodes.items():
        registry.get_or_create(ilat, ilon, name=iid)

    segments: list[AirwaySegment] = []
    added = skipped = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        line_geoms = (
            [geom]           if geom.geom_type == "LineString"       else
            list(geom.geoms) if geom.geom_type == "MultiLineString"  else []
        )
        if not line_geoms:
            skipped += 1
            continue

        kv = float(row[kv_col]) if kv_col and not pd.isna(row.get(kv_col, float("nan"))) else 0.0
        airway_id = _voltage_label(kv)
        sub1  = _clean_name(row[sub1_col])  if sub1_col  else ""
        sub2  = _clean_name(row[sub2_col])  if sub2_col  else ""
        state = str(row[state_col]).strip()[:2] if state_col else ""

        for line in line_geoms:
            raw_coords = list(line.coords)
            if len(raw_coords) < 2:
                continue

            # Re-run interpolation so weather provider is sampled at fine
            # enough spatial resolution (same as original path).
            pts = _interpolate_line(raw_coords, max_segment_nm)

            node_idents: list[str] = []
            for i, (lon, lat) in enumerate(pts):
                key = (round(lon, 5), round(lat, 5))
                if key in xing_nodes:
                    # Intersection node — use its stable UUID ident
                    ident = xing_nodes[key]
                    registry.get_or_create(lat, lon, name=ident)
                    node_idents.append(ident)
                else:
                    name  = sub1 if i == 0 else (sub2 if i == len(pts) - 1 else "")
                    ident = registry.get_or_create(lat, lon, name=name, state=state)
                    node_idents.append(ident)

            # Collapse consecutive duplicates (can arise after snapping)
            deduped = [node_idents[0]]
            for nid in node_idents[1:]:
                if nid != deduped[-1]:
                    deduped.append(nid)

            for i in range(len(deduped) - 1):
                segments.append(AirwaySegment(
                    airway_id=airway_id,
                    from_ident=deduped[i],
                    to_ident=deduped[i + 1],
                    seq_from=i,
                    seq_to=i + 1,
                    voltage_kv=kv,
                ))
                added += 1

    fixes = registry.fixes
    print(f"  [TX]  {len(fixes):,} nodes, {added:,} segments "
          f"({skipped} bad-geometry skipped)")
    print(f"\n  [TX]  Graph ready (pre-processed): {len(fixes):,} nodes, "
          f"{len(segments):,} segments\n")

    return GraphData(fixes=fixes, segments=segments, source="transmission")


# ── Main loader ───────────────────────────────────────────────────────────────

def load_transmission(
    eia_dir: str | Path = "data/eia",
    voltage_min:       float = MIN_VOLTAGE_KV,
    max_segment_nm:    float = MAX_SEGMENT_NM,
    snap_threshold_km: float = SNAP_THRESHOLD_KM,
    bbox: tuple | None = None,
    preprocessed: "str | Path | bool | None" = None,
) -> GraphData:
    """
    Build a GraphData from the EIA transmission lines shapefile.

    If a pre-processed GeoPackage exists (written by preprocess_tx.py),
    it is used instead of the raw shapefile.  Pass preprocessed=False to
    force the raw path even when the cache file is present.

    Parameters
    ----------
    eia_dir          : directory containing the unzipped shapefile.
                       Auto-detects .shp or .gpkg one level deep.
    voltage_min      : exclude lines below this kV (default 115)
    max_segment_nm   : split segments longer than this (default 25 nm)
    snap_threshold_km: merge endpoint nodes within this distance (default 1 km)
    bbox             : (min_lon, min_lat, max_lon, max_lat) bounding box.
                       Lines outside this box are dropped before processing.
                       Passed directly to pyogrio so a 42k-feature national
                       file reads only the spatial subset from disk.
                       Default covers ORF→CRW AO (VA/WV/MD/NC + 1° pad).
    preprocessed     : path to pre-processed GeoPackage, or None to
                       auto-detect data/eia/tx_preprocessed.gpkg, or False
                       to skip the cache and always read raw data.

    Returns
    -------
    GraphData with source="transmission"
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
    except ImportError as exc:
        raise ImportError("geopandas and shapely are required: pip install geopandas shapely") from exc

    # ── Pre-processed cache check ─────────────────────────────────────────────
    eia_dir = Path(eia_dir)

    if preprocessed is False:
        use_cache  = False
        cache_path = None
    elif preprocessed is not None:
        cache_path = Path(preprocessed)
        use_cache  = cache_path.exists()
    else:
        cache_path = eia_dir / "tx_preprocessed.gpkg"
        use_cache  = cache_path.exists()

    if use_cache:
        return _load_preprocessed(cache_path, max_segment_nm, snap_threshold_km)

    # ── Raw shapefile path (unchanged from original) ──────────────────────────

    # Default: ORF→CRW AO — VA, WV, MD, NC with 1° buffer
    if bbox is None:
        bbox = (-83.5, 36.0, -75.5, 39.5)

    shp_path = _find_shapefile(eia_dir)

    print(f"\n[TX Parser] Loading: {shp_path.name}")
    print(f"  [TX]  Bbox (WGS84): lon {bbox[0]}..{bbox[2]}  lat {bbox[1]}..{bbox[3]}")

    # Read CRS without loading data, then transform bbox to match the file CRS.
    # pyogrio interprets bbox in the file's native CRS, so passing WGS84
    # degrees to a projected (metre-based) file returns 0 features.
    from pyproj import Transformer
    file_crs = gpd.read_file(shp_path, engine="pyogrio", rows=0).crs
    if file_crs is not None and file_crs.to_epsg() != 4326:
        print(f"  [TX]  File CRS: EPSG:{file_crs.to_epsg()} — transforming bbox")
        t = Transformer.from_crs("EPSG:4326", file_crs, always_xy=True)
        x0, y0 = t.transform(bbox[0], bbox[1])
        x1, y1 = t.transform(bbox[2], bbox[3])
        read_bbox = (x0, y0, x1, y1)
    else:
        read_bbox = bbox

    gdf = gpd.read_file(shp_path, engine="pyogrio", bbox=read_bbox)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Precise clip in WGS84 (bbox read filter uses envelope, not exact geometry)
    clip_geom = shapely_box(*bbox)
    gdf = gdf[gdf.geometry.intersects(clip_geom)].copy()
    print(f"  [TX]  {len(gdf):,} features in AO")


    # ── Column detection ──────────────────────────────────────────────────────
    volt_col   = _get_col(gdf, ["VOLTAGE",    "voltage",   "Voltage",  "VOLT"])
    vclass_col = _get_col(gdf, ["VOLT_CLASS", "voltclass", "VOLTCLASS"])
    status_col = _get_col(gdf, ["STATUS",     "status"])
    sub1_col   = _get_col(gdf, ["SUB_1",  "sub_1",  "FROM_SUB", "FROMSUB", "from_sub"])
    sub2_col   = _get_col(gdf, ["SUB_2",  "sub_2",  "TO_SUB",   "TOSUB",   "to_sub"])
    state_col  = _get_col(gdf, ["STATE",  "state",  "STATE_NAME"])

    print(f"  [TX]  Detected — voltage:{volt_col}  volt_class:{vclass_col}  "
          f"status:{status_col}  sub1:{sub1_col}  sub2:{sub2_col}")

    # ── Status filter ─────────────────────────────────────────────────────────
    if status_col:
        mask = gdf[status_col].str.upper().str.contains(
            r"IN.?SERVICE|ENERGIZED|OPERATIONAL", na=True
        )
        gdf = gdf[mask].copy()
        print(f"  [TX]  {len(gdf):,} in-service lines after status filter")

    # ── Voltage filter ────────────────────────────────────────────────────────
    kv_series = _numeric_voltage(gdf, volt_col, vclass_col)
    gdf = gdf[kv_series.fillna(0) >= voltage_min].copy()
    gdf["_kv"] = kv_series[gdf.index]
    print(f"  [TX]  {len(gdf):,} lines at ≥{voltage_min:.0f} kV")

    if len(gdf) == 0:
        print("  [TX]  WARNING: no lines passed voltage filter — check column names above")
        return GraphData(fixes={}, segments=[], source="transmission")

    # ── Build graph ───────────────────────────────────────────────────────────
    registry = _NodeRegistry(snap_threshold_km)
    segments: list[AirwaySegment] = []
    added = skipped = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        kv   = float(row["_kv"]) if not pd.isna(row["_kv"]) else 0.0

        if geom is None or geom.is_empty:
            skipped += 1
            continue

        line_geoms = (
            [geom]          if geom.geom_type == "LineString"      else
            list(geom.geoms) if geom.geom_type == "MultiLineString" else []
        )
        if not line_geoms:
            skipped += 1
            continue

        sub1  = _clean_name(row[sub1_col])  if sub1_col  else ""
        sub2  = _clean_name(row[sub2_col])  if sub2_col  else ""
        state = str(row[state_col]).strip()[:2] if state_col else ""
        airway_id = _voltage_label(kv)

        for line in line_geoms:
            raw_coords = list(line.coords)   # (lon, lat)
            if len(raw_coords) < 2:
                continue

            pts = _interpolate_line(raw_coords, max_segment_nm)

            # Assign node idents; first and last points use SUB names
            node_idents: list[str] = []
            for i, (lon, lat) in enumerate(pts):
                name  = sub1 if i == 0 else (sub2 if i == len(pts) - 1 else "")
                ident = registry.get_or_create(lat, lon, name=name, state=state)
                node_idents.append(ident)

            # Collapse consecutive duplicates (can arise after snapping)
            deduped = [node_idents[0]]
            for nid in node_idents[1:]:
                if nid != deduped[-1]:
                    deduped.append(nid)

            for i in range(len(deduped) - 1):
                segments.append(AirwaySegment(
                    airway_id=airway_id,
                    from_ident=deduped[i],
                    to_ident=deduped[i + 1],
                    seq_from=i,
                    seq_to=i + 1,
                    voltage_kv=kv,
                ))
                added += 1

    fixes = registry.fixes
    print(f"  [TX]  {len(fixes):,} nodes, {added:,} segments "
          f"({skipped} bad-geometry skipped)")
    print(f"\n  [TX]  Graph ready: {len(fixes):,} nodes, {len(segments):,} segments\n")

    return GraphData(fixes=fixes, segments=segments, source="transmission")
