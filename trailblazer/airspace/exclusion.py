"""
trailblazer/airspace/exclusion.py — Airspace classification and exclusion

Uses authoritative FAA polygon boundaries from the NASR class airspace shapefiles.
Falls back to radius heuristics only if shapefiles are not present.

Download (authoritative — no guesswork)
─────────────────────────────────────────
1. Go to https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
2. Select the current 28-day cycle
3. Download "Shape File Data" (class_airspace_shape_files.zip)
4. Unzip into data/nasr/class_airspace/

Expected layout after unzip:
    data/nasr/class_airspace/
        Class_Airspace.shp      ← single file with CLASS field
    OR
        Class_B.shp
        Class_C.shp
        Class_D.shp
        Class_E_Surface.shp     ← surface-level Class E (e.g. around non-towered airports)

The shapefile has a CLASS field: "B", "C", "D", "E" etc.
NASR shapefiles also carry LOWER_VAL / LOWER_DESC altitude fields used by the
altitude filter — see _load_shape_gdf().
GeoPandas does the point-in-polygon queries; no radius approximation needed.

TFR injection
─────────────
Edit data/tfrs.json to define synthetic TFRs. Format:
    [{"name": "...", "polygon": [[lat,lon],...], "start_utc": "...", "end_utc": "..."}]

A sample file is written on first run if none exists.

Weather zone injection
──────────────────────
Edit data/weather_zones.json to define manual weather exclusion zones. Format:
    [{"name": "...", "polygon": [[lat,lon],...],
      "severity": "IFR", "hazard": "IFR",
      "start_utc": "...", "end_utc": "..."}]

Weather zones are structurally parallel to TFRs but kept separate so airspace
and weather exclusions can evolve independently. G-AIRMET advisories above a
configurable severity threshold are converted to WeatherZone objects at runtime
by the Streamlit app and passed to apply_weights() alongside any manually
injected zones.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Airspace class constants ───────────────────────────────────────────────────

AIRSPACE_CLASSES = ("G", "E", "D", "C", "B", "TFR", "SFRA", "WX", "UNKNOWN")

# IBM colorblind palette for airspace overlays
AIRSPACE_COLORS = {
    "G":    None,
    "E":    "#785EF0",   # violet — semi-transparent overlay
    "D":    "#FE6100",   # orange
    "C":    "#FFB000",   # gold
    "B":    "#DC267F",   # magenta
    "TFR":  "#FA4D56",   # red
    "SFRA": "#FA4D56",   # red
    "WX":   "#009D9A",   # teal — weather exclusion zone
}

# DC SFRA — always hard exclusion regardless of shapefile coverage
_KDCA_LAT, _KDCA_LON = 38.8521, -77.0377
_SFRA_RADIUS_NM = 15.0

# Fallback radius buffers (used only when shapefiles are unavailable)
# OurAirports type → (airspace_class, radius_nm)
_FALLBACK_BUFFERS = {
    "large_airport":  ("C", 10.0),   # most large_airports in US are Class C
    "medium_airport": ("D",  4.0),
    "small_airport":  ("D",  4.0),
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TFR:
    name:       str
    polygon:    list[tuple[float, float]]   # (lat, lon) pairs
    start_utc:  Optional[datetime] = None
    end_utc:    Optional[datetime] = None


@dataclass
class WeatherZone:
    """
    A weather-based spatial exclusion zone.  Structurally parallel to TFR but
    kept separate so airspace and weather exclusions evolve independently.

    Sources:
        - Manually injected events in data/weather_zones.json
        - G-AIRMET advisories converted by streamlit_app.py above a severity
          threshold (controlled by the sidebar slider)

    Fields
    ──────
    severity : "MVFR" | "IFR" | "LIFR"
               The worst wx category within the zone.  apply_weights() compares
               this against wx_zone_severity_threshold to decide whether the zone
               is a hard exclusion (weight=inf).
    hazard   : Informational only (e.g. "IFR", "MTN OBSCN", "TURB", "ICE").
               Does not drive routing behaviour — severity does.
    """
    name:      str
    polygon:   list[tuple[float, float]]   # (lat, lon) pairs
    severity:  str                          # "MVFR" | "IFR" | "LIFR"
    hazard:    str = ""                     # informational
    start_utc: Optional[datetime] = None
    end_utc:   Optional[datetime] = None


@dataclass
class AirspaceExclusion:
    """
    Spatial lookup for airspace classification at a point.

    Loads authoritative FAA polygon boundaries from NASR shapefiles when
    available. Falls back to simplified radius buffers otherwise.

    Call classify_point() for each edge midpoint during graph construction.
    """
    tfrs:          list[TFR]  = field(default_factory=list)
    _use_shapefile: bool      = field(default=False, repr=False)
    _gdf:          object     = field(default=None,  repr=False)   # GeoDataFrame or None
    _fallback:     list[dict] = field(default_factory=list, repr=False)

    @property
    def zones(self) -> list[dict]:
        """Fallback radius zones for map display (used when shapefiles absent)."""
        return self._fallback

    def classify_point(
        self,
        lat: float,
        lon: float,
        t:   Optional[datetime] = None,
    ) -> str:
        """
        Return the most restrictive airspace class at (lat, lon) at time t.
        TFRs are checked first; then SFRA; then FAA polygon / fallback radii.
        """
        # TFRs
        for tfr in self.tfrs:
            if t is not None:
                t_utc = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
                if tfr.start_utc and t_utc < tfr.start_utc:
                    continue
                if tfr.end_utc and t_utc > tfr.end_utc:
                    continue
            if _ray_cast(lat, lon, tfr.polygon):
                return "TFR"

        # DC SFRA
        if _haversine_nm(_KDCA_LAT, _KDCA_LON, lat, lon) <= _SFRA_RADIUS_NM:
            return "SFRA"

        # FAA polygon lookup
        if self._use_shapefile and self._gdf is not None:
            return _classify_from_gdf(self._gdf, lat, lon)

        # Fallback: radius buffers
        _rank = {"G": 0, "E": 1, "D": 2, "C": 3, "B": 4}
        worst = "G"
        for z in self._fallback:
            if _haversine_nm(z["lat"], z["lon"], lat, lon) <= z["radius_nm"]:
                cls = z["airspace"]
                if _rank.get(cls, 0) > _rank.get(worst, 0):
                    worst = cls
        return worst

    def classify_segment(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        t:    Optional[datetime] = None,
    ) -> str:
        """
        Return the most restrictive airspace class intersected by the segment
        (lat1,lon1)→(lat2,lon2) at time t.

        Uses shapely LineString intersection against FAA polygon boundaries so
        that a segment is correctly classified even when its midpoint falls
        outside the airspace polygon.  Falls back to midpoint classify_point
        when shapely is unavailable or shapefiles are not loaded.

        This is the correct primitive to call during graph construction —
        classify_point only catches edges whose midpoint happens to lie inside
        a polygon, missing any edge that enters and exits an airspace region
        with its midpoint outside it.
        """
        _rank = {"G": 0, "E": 1, "D": 2, "C": 3, "B": 4, "TFR": 5, "SFRA": 6}
        worst = "G"

        # ── Time gate (shared by all checks below) ────────────────────────────
        t_utc = None
        if t is not None:
            t_utc = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)

        # ── Build shapely segment (lon, lat order for EPSG:4326) ──────────────
        try:
            from shapely.geometry import LineString as _LS, Point as _Pt
            seg = _LS([(lon1, lat1), (lon2, lat2)])
            _have_shapely = True
        except ImportError:
            _have_shapely = False
            seg = None

        # ── TFRs ──────────────────────────────────────────────────────────────
        for tfr in self.tfrs:
            if t_utc is not None:
                if tfr.start_utc and t_utc < tfr.start_utc:
                    continue
                if tfr.end_utc   and t_utc > tfr.end_utc:
                    continue
            if _have_shapely:
                try:
                    from shapely.geometry import Polygon as _Poly
                    tfr_poly = _Poly([(p[1], p[0]) for p in tfr.polygon])
                    if seg.intersects(tfr_poly):
                        return "TFR"
                except Exception:
                    pass
            else:
                mid_lat = (lat1 + lat2) / 2
                mid_lon = (lon1 + lon2) / 2
                if any(_ray_cast(la, lo, tfr.polygon)
                       for la, lo in [(lat1, lon1), (lat2, lon2), (mid_lat, mid_lon)]):
                    return "TFR"

        # ── DC SFRA (circular buffer — check min distance from segment) ───────
        if _have_shapely:
            try:
                dca_pt  = _Pt(_KDCA_LON, _KDCA_LAT)
                deg_tol = _SFRA_RADIUS_NM / 60.0
                if seg.distance(dca_pt) <= deg_tol:
                    worst = "SFRA"
            except Exception:
                pass
        else:
            for la, lo in [(lat1, lon1), (lat2, lon2), ((lat1+lat2)/2, (lon1+lon2)/2)]:
                if _haversine_nm(_KDCA_LAT, _KDCA_LON, la, lo) <= _SFRA_RADIUS_NM:
                    worst = "SFRA"
                    break

        if _rank.get(worst, 0) >= _rank.get("SFRA", 0):
            return worst

        # ── FAA shapefile: segment intersection ───────────────────────────────
        if self._use_shapefile and self._gdf is not None and _have_shapely:
            cls = _classify_segment_from_gdf(self._gdf, seg)
            if _rank.get(cls, 0) > _rank.get(worst, 0):
                worst = cls
        elif self._use_shapefile and self._gdf is not None:
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            cls = _classify_from_gdf(self._gdf, mid_lat, mid_lon)
            if _rank.get(cls, 0) > _rank.get(worst, 0):
                worst = cls
        else:
            for la, lo in [(lat1, lon1), (lat2, lon2), ((lat1+lat2)/2, (lon1+lon2)/2)]:
                for z in self._fallback:
                    if _haversine_nm(z["lat"], z["lon"], la, lo) <= z["radius_nm"]:
                        cls = z["airspace"]
                        if _rank.get(cls, 0) > _rank.get(worst, 0):
                            worst = cls

        return worst


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_airspace(
    nasr_shape_dir:  str | Path | None = None,
    airports_csv:    str | Path | None = None,
    tfrs_json:       str | Path | None = None,
    bbox:            tuple | None      = None,
    operating_agl_ft: float            = 400.0,
) -> AirspaceExclusion:
    """
    Build an AirspaceExclusion.

    nasr_shape_dir   : directory containing class_airspace shapefiles.
                       Auto-detected at data/nasr/class_airspace/.
    airports_csv     : OurAirports CSV (fallback only). Auto-detected.
    tfrs_json        : TFR JSON. Auto-detected at data/tfrs.json.
    bbox             : (min_lon, min_lat, max_lon, max_lat) clip.
    operating_agl_ft : Operating altitude in feet AGL.  Used to filter Class E
                       shapefiles — only surface-level and sub-ceiling Class E
                       polygons are retained.  Default 400 ft (Part 107 ceiling).
                       Pass 350 for this mission profile.
    """
    tfrs  = _load_tfrs(tfrs_json)
    ex    = AirspaceExclusion(tfrs=tfrs)

    # Try FAA shapefiles first
    shape_dir = _find_shape_dir(nasr_shape_dir)
    if shape_dir:
        gdf = _load_shape_gdf(shape_dir, bbox, operating_agl_ft=operating_agl_ft)
        if gdf is not None and len(gdf) > 0:
            ex._use_shapefile = True
            ex._gdf           = gdf
            print(f"  [Airspace] FAA shapefiles: {len(gdf)} polygons loaded "
                  f"(operating alt {operating_agl_ft:.0f} ft AGL)")
        else:
            print("  [Airspace] FAA shapefiles found but empty — using fallback")
    else:
        print("  [Airspace] No NASR shapefiles found — using radius fallback")
        print("             Download: NASR subscription → Shape File Data")
        print("             → class_airspace_shape_files.zip → data/nasr/class_airspace/")

    # Fallback: radius buffers from airports CSV
    if not ex._use_shapefile:
        ex._fallback = _load_fallback_zones(airports_csv, bbox)
        print(f"  [Airspace] Fallback: {len(ex._fallback)} airport radius zones")

    print(f"  [Airspace] {len(tfrs)} TFR(s) loaded")
    return ex


def load_weather_zones(json_path=None) -> list[WeatherZone]:
    """
    Load manually-injected weather exclusion zones from data/weather_zones.json.

    Returns an empty list if the file does not exist — no crash, no sample write.
    Weather zones are created explicitly by the operator (via sidebar inject or
    direct file edit), unlike TFRs which auto-create a demo on first run.

    G-AIRMET-derived zones are NOT loaded here — they are constructed at runtime
    from the G-AIRMET cache by the Streamlit app and merged with these manual zones
    before being passed to apply_weights().
    """
    zones = _load_weather_zones(json_path)
    print(f"  [Weather] {len(zones)} manual weather zone(s) loaded")
    return zones


# ── FAA shapefile loading ──────────────────────────────────────────────────────

def _find_shape_dir(explicit: str | Path | None) -> Optional[Path]:
    """Locate the class airspace shapefile directory."""
    candidates = [
        explicit,
        "data/nasr/class_airspace",
        "data/nasr/Class_Airspace",
    ]
    for p in candidates:
        if p and Path(p).exists() and Path(p).is_dir():
            contents = list(Path(p).glob("*.shp"))
            if contents:
                return Path(p)
    return None


def _load_shape_gdf(shape_dir: Path, bbox: tuple | None, operating_agl_ft: float = 400.0):
    """
    Load class airspace polygons into a GeoDataFrame.

    Altitude filter (Class E only)
    ────────────────────────────────
    NASR shapefiles carry LOWER_VAL (numeric, ft) and LOWER_DESC ("SFC", "AGL", "MSL")
    for each polygon.  At a typical BVLOS operating altitude of 350–400 ft AGL the
    drone is in Class G under en-route Class E (floor 1,200 ft AGL) and transition
    areas (floor 700 ft AGL).  Only surface-level Class E — floor = "SFC" or
    LOWER_VAL ≤ operating_agl_ft — is relevant and retained.

    If the shapefile lacks LOWER_VAL / LOWER_DESC columns (older NASR cycles or
    split-class files), all Class E polygons are retained with a warning.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
    except ImportError:
        print("  [Airspace] geopandas not installed — cannot load shapefiles")
        return None

    shp_files = list(shape_dir.glob("*.shp"))
    if not shp_files:
        return None

    gdfs = []
    for shp in shp_files:
        try:
            gdf = gpd.read_file(shp, engine="pyogrio")
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            gdfs.append(gdf)
        except Exception as exc:
            print(f"  [Airspace] Could not read {shp.name}: {exc}")

    if not gdfs:
        return None

    import pandas as pd
    combined = pd.concat(gdfs, ignore_index=True)
    import geopandas as gpd
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    # Normalise CLASS field
    cls_col = _find_col(combined, ["CLASS", "class", "AirspaceClass", "AIRSPACE_CL"])
    if cls_col is None:
        print("  [Airspace] WARNING: no CLASS column found in shapefile")
        return None

    combined["_class"] = combined[cls_col].str.strip().str.upper()

    # Airport identifier — used for map tooltips (e.g. "KORF · Class C")
    ident_col = _find_col(combined, ["IDENT", "ident", "FAA_CD", "ARPT_IDENT",
                                     "ARPT_ID", "ICAO", "icao", "ID"])
    combined["_ident"] = (
        combined[ident_col].str.strip().str.upper()
        if ident_col else ""
    )

    # Clip to bbox
    if bbox:
        clip = shapely_box(*bbox)
        combined = combined[combined.geometry.intersects(clip)].copy()

    # ── Altitude filter for Class E ───────────────────────────────────────────
    # At operating_agl_ft (e.g. 350 ft), the drone is below en-route Class E
    # (floor 1,200 ft AGL) and transition areas (floor 700 ft AGL).  Only
    # surface-level Class E is controlled airspace at this altitude.
    #
    # LOWER_DESC == "SFC"                 → always keep (surface Class E)
    # LOWER_VAL  <= operating_agl_ft      → keep (floor is below us)
    # Everything else                     → drop (drone is in Class G under it)
    is_class_e = combined["_class"] == "E"

    if is_class_e.any():
        lower_desc_col = _find_col(combined, ["LOWER_DESC", "lower_desc", "LOWERDESC"])
        lower_val_col  = _find_col(combined, ["LOWER_VAL",  "lower_val",  "LOWERVAL"])

        if lower_desc_col is not None or lower_val_col is not None:
            # Build a boolean mask: True = this Class E row should be kept
            keep_e = pd.Series(False, index=combined.index)

            if lower_desc_col is not None:
                desc = combined[lower_desc_col].fillna("").str.strip().str.upper()
                keep_e |= is_class_e & (desc == "SFC")

            if lower_val_col is not None:
                vals = pd.to_numeric(combined[lower_val_col], errors="coerce").fillna(float("inf"))
                keep_e |= is_class_e & (vals <= operating_agl_ft)

            n_before  = int(is_class_e.sum())
            n_kept_e  = int((is_class_e & keep_e).sum())
            n_dropped = n_before - n_kept_e

            # Keep all non-E rows, plus only the surface-level E rows
            combined = combined[~is_class_e | keep_e].copy()

            print(f"  [Airspace] Class E altitude filter ({operating_agl_ft:.0f} ft AGL): "
                  f"kept {n_kept_e}/{n_before} polygons, dropped {n_dropped} above operating altitude")
        else:
            print(f"  [Airspace] WARNING: no LOWER_VAL/LOWER_DESC in shapefile — "
                  f"all {int(is_class_e.sum())} Class E polygon(s) retained "
                  f"(may over-report Class E at {operating_agl_ft:.0f} ft AGL)")

    # Keep only B/C/D/E (surface-filtered)
    combined = combined[combined["_class"].isin(["B", "C", "D", "E"])]
    return combined


def _classify_from_gdf(gdf, lat: float, lon: float) -> str:
    """
    Point-in-polygon against the FAA class airspace GeoDataFrame.
    Uses the GeoDataFrame spatial index (STRtree) for fast lookup —
    O(log n) instead of O(n) linear scan over all polygons.
    """
    from shapely.geometry import Point
    pt = Point(lon, lat)
    _rank = {"B": 4, "C": 3, "D": 2, "E": 1}
    worst = "G"

    try:
        candidates = list(gdf.sindex.query(pt, predicate="contains"))
    except Exception:
        candidates = range(len(gdf))

    for idx in candidates:
        try:
            row = gdf.iloc[idx]
            cls = row["_class"]
            if _rank.get(cls, 0) > _rank.get(worst, 0):
                worst = cls
        except Exception:
            continue
    return worst


def _classify_segment_from_gdf(gdf, seg) -> str:
    """
    Return the most restrictive airspace class whose polygon intersects `seg`
    (a shapely LineString in EPSG:4326 lon/lat).

    Uses the GeoDataFrame spatial index for O(log n) candidate selection, then
    exact shapely intersection for correctness.  Replaces the point-in-polygon
    midpoint check so that segments crossing a polygon boundary are caught even
    when the midpoint falls outside.
    """
    _rank = {"B": 4, "C": 3, "D": 2, "E": 1}
    worst = "G"
    try:
        candidates = list(gdf.sindex.query(seg, predicate="intersects"))
    except Exception:
        candidates = range(len(gdf))
    for idx in candidates:
        try:
            row = gdf.iloc[idx]
            cls = row["_class"]
            if _rank.get(cls, 0) > _rank.get(worst, 0):
                if seg.intersects(row.geometry):
                    worst = cls
        except Exception:
            continue
    return worst


def _find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


# ── Fallback radius zones ──────────────────────────────────────────────────────

def _load_fallback_zones(csv_path, bbox) -> list[dict]:
    """Load simplified airport radius buffers from OurAirports CSV."""
    import csv as csv_mod

    candidates = [csv_path, "data/airports.csv",
                  "../weatherboy/config/maps/airports.csv"]
    found = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if not found:
        return []

    buf_deg = 0.5
    if bbox:
        min_lon, min_lat, max_lon, max_lat = (
            bbox[0] - buf_deg, bbox[1] - buf_deg,
            bbox[2] + buf_deg, bbox[3] + buf_deg,
        )

    zones = []
    with open(found, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            if row.get("iso_country", "").strip().upper() != "US":
                continue
            atype = row.get("type", "").strip().lower()
            if atype not in _FALLBACK_BUFFERS:
                continue
            try:
                lat = float(row["latitude_deg"])
                lon = float(row["longitude_deg"])
            except (ValueError, KeyError):
                continue
            if bbox and not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            airspace, radius = _FALLBACK_BUFFERS[atype]
            if airspace == "D" and row.get("scheduled_service", "no").strip() != "yes":
                continue
            zones.append({"lat": lat, "lon": lon,
                          "airspace": airspace, "radius_nm": radius,
                          "ident": row.get("ident", "")})
    return zones


# ── TFR loading ────────────────────────────────────────────────────────────────

def _load_tfrs(json_path) -> list[TFR]:
    candidates = [json_path, "data/tfrs.json"]
    found = next((Path(p) for p in candidates if p and Path(p).exists()), None)

    if found is None:
        _write_sample_tfrs(Path("data/tfrs.json"))
        return []

    try:
        raw = json.loads(found.read_text())
    except Exception as exc:
        print(f"  [Airspace] Could not parse TFR file: {exc}")
        return []

    tfrs = []
    for item in raw:
        try:
            poly = [(float(pt[0]), float(pt[1])) for pt in item["polygon"]]
            if len(poly) < 3:
                continue

            def _pt(s):
                return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None

            tfrs.append(TFR(
                name=item.get("name", "TFR"),
                polygon=poly,
                start_utc=_pt(item.get("start_utc")),
                end_utc=_pt(item.get("end_utc")),
            ))
        except (KeyError, IndexError, ValueError) as exc:
            print(f"  [Airspace] TFR parse error: {exc}")

    return tfrs


def _write_sample_tfrs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = [{
        "name": "DEMO TFR ALPHA — Shenandoah Valley",
        "polygon": [[38.80,-78.70],[38.80,-78.20],[38.40,-78.20],[38.40,-78.70]],
        "start_utc": "2026-05-14T14:00:00Z",
        "end_utc":   "2026-05-14T20:00:00Z",
    }]
    path.write_text(json.dumps(sample, indent=2))
    print(f"  [Airspace] Sample TFR written to {path}")


# ── Weather zone loading ───────────────────────────────────────────────────────

def _load_weather_zones(json_path) -> list[WeatherZone]:
    """Parse data/weather_zones.json into WeatherZone objects."""
    candidates = [json_path, "data/weather_zones.json"]
    found = next((Path(p) for p in candidates if p and Path(p).exists()), None)

    if found is None:
        return []   # no sample written — zones are always intentional

    try:
        raw = json.loads(found.read_text())
    except Exception as exc:
        print(f"  [Weather] Could not parse weather zones file: {exc}")
        return []

    zones = []
    for item in raw:
        try:
            poly = [(float(pt[0]), float(pt[1])) for pt in item["polygon"]]
            if len(poly) < 3:
                continue

            def _pt(s):
                return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None

            zones.append(WeatherZone(
                name=item.get("name", "Weather Zone"),
                polygon=poly,
                severity=item.get("severity", "IFR").upper(),
                hazard=item.get("hazard", ""),
                start_utc=_pt(item.get("start_utc")),
                end_utc=_pt(item.get("end_utc")),
            ))
        except (KeyError, IndexError, ValueError) as exc:
            print(f"  [Weather] Zone parse error: {exc}")

    return zones


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _ray_cast(lat: float, lon: float, ring: list[tuple]) -> bool:
    """Point-in-polygon ray cast. ring = [(lat,lon), ...]"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        yi, xi = ring[i]
        yj, xj = ring[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside