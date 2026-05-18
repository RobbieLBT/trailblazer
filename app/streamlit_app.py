"""
app/streamlit_app.py — Trailblazer interactive route planner

Sliders re-weight the graph in real time without rebuilding it.
G-AIRMET layer fetched from local cache and shown as a prog-chart-style overlay.
TFR polygons loaded from data/tfrs.json (synthetic — edit to inject events).
Weather zones loaded from data/weather_zones.json and/or converted from
G-AIRMET advisories above a configurable severity threshold.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent))

from trailblazer.routing.pathfinder import (
    apply_weights, find_routes, resolve_endpoint,
    AIRSPACE_PENALTY,
)
from trailblazer.airspace.exclusion import (
    AirspaceExclusion, WeatherZone, AIRSPACE_COLORS, _ray_cast, _haversine_nm,
    load_airspace, load_weather_zones,
)

# ── IBM colorblind-safe palette ───────────────────────────────────────────────

IBM = {
    "blue":    "#648FFF",
    "violet":  "#785EF0",
    "magenta": "#DC267F",
    "orange":  "#FE6100",
    "gold":    "#FFB000",
    "teal":    "#009D9A",
    "cyan":    "#33B1FF",
    "green":   "#198038",
    "gray":    "#A8B2BD",
    "red":     "#FA4D56",
}

# Network tile colour — all edges use this single hue; voltage encoded by
# lineweight in tiles.py.  Keep in sync with tiles._EDGE_COLOR.
_NETWORK_COLOR = IBM["cyan"]

# VOLTAGE_COLORS kept for any fallback GeoJSON path; all set to network colour
# so the legend can drive lineweight display without a colour mismatch.
VOLTAGE_COLORS = {
    "765KV":      _NETWORK_COLOR,
    "500KV":      _NETWORK_COLOR,
    "345KV":      _NETWORK_COLOR,
    "230KV":      _NETWORK_COLOR,
    "115KV":      _NETWORK_COLOR,
    "TXLOW":      _NETWORK_COLOR,
    "TOWER_MESH": _NETWORK_COLOR,
}

# Lineweights for legend swatches (px height) — must mirror tiles._BASE_LW
_VOLTAGE_LW_PX: dict[str, int] = {
    "765KV": 4, "500KV": 3, "345KV": 3, "230KV": 2, "115KV": 2,
    "TOWER_MESH": 1,
}

ROUTE_COLORS = [IBM["blue"], IBM["orange"], IBM["green"], IBM["magenta"], IBM["violet"]]

WX_COLORS = {
    "VFR":  None,
    "MVFR": "#648FFF",
    "IFR":  "#DC267F",
    "LIFR": "#785EF0",
}

# Open-Topo-Data SRTM 30m — used for live elevation profile fetch
_ELEV_API_URL  = "https://api.opentopodata.org/v1/srtm30m"
_ELEV_BATCH    = 100    # hard limit per request
_ELEV_DELAY    = 1.1    # seconds between requests (public instance rate limit)
_ELEV_SAMPLES  = 6      # sample points per graph segment


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Trailblazer", page_icon="🛰️", layout="wide")
st.title("🛰️ Trailblazer — BVLOS Route Planner")

# ── Graph loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


pkl_options = sorted(Path(".").glob("graph_*.pkl"))
if not pkl_options:
    st.error("No graph_*.pkl found. Run: `python build_graph.py`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Graph")
    pkl_choice = st.selectbox("File", pkl_options, format_func=lambda p: p.name)

G, graph_data = load_pkl(pkl_choice)
has_elev_data_pre = any(f.elevation_m is not None for f in graph_data.fixes.values())

with st.sidebar:
    st.header("🗺️ Route")
    origin_input = st.text_input("Origin",      "ORF",
                                 help="ICAO, lat,lon, or node ident")
    dest_input   = st.text_input("Destination", "CRW",
                                 help="ICAO, lat,lon, or node ident")
    cruise_kts   = st.slider("Cruise speed (kts)", 60, 200, 120, step=10)
    k_routes     = st.slider("Routes (k)", 1, 5, 3)

    st.header("⚖️ Objective weights")

    time_weight = st.slider("Time  λ_t", 0.0, 2.0, 1.0, 0.1,
                            help="Weight on ETE. 1.0 = pure min-time baseline.")

    has_elev_data = has_elev_data_pre
    elev_label = "Altitude  λ_e" if has_elev_data else "Altitude  λ_e ⚠️ no data"
    elev_help = ("Penalty per metre of climb. 0 = pure time routing."
                 if has_elev_data else
                 "⚠️ Elevation data not loaded — rebuild with "
                 "`python build_graph.py --elevation` to activate.")
    elev_weight = st.slider(elev_label, 0.0, 0.5, 0.0, 0.01,
                            format="%.2f",
                            disabled=not has_elev_data,
                            help=elev_help)

    noise_weight = st.slider("Noise  λ_n", 0.0, 1.0, 0.0, 0.05,
                             help="⚠️ Pending LandScan data. Slider wired; "
                                  "score = 0 until Phase 4.")
    if noise_weight > 0:
        st.caption("🔶 Noise scoring requires LandScan USA raster (Phase 4). "
                   "Routes unchanged until data is loaded.")

    st.header("🌤️ Weather")
    wx_filter = st.select_slider(
        "Max wx category",
        options=["VFR", "MVFR", "IFR", "ALL"],
        value="MVFR",
        help="Edges with baked wx_rank worse than this are excluded from routing.",
    )
    show_gairmet = st.checkbox("Show G-AIRMET overlay", value=True)
    if show_gairmet and not Path("data/gairmets.cache.xml.gz").exists():
        st.caption("📥 Download cache: [gairmets.cache.xml.gz]"
                   "(https://aviationweather.gov/data/cache/gairmets.cache.xml.gz)")
        st.caption("Save to `data/gairmets.cache.xml.gz`")

    apply_gairmet_routing = st.checkbox(
        "Apply G-AIRMETs to routing", value=True,
        help="Convert G-AIRMET advisories at or above the threshold below into "
             "hard exclusion zones (same mechanism as TFRs). Takes effect "
             "immediately — no graph rebuild needed.",
    )
    wx_zone_threshold = st.select_slider(
        "G-AIRMET / weather zone exclusion threshold",
        options=["MVFR", "IFR"],
        value="IFR",
        help="Advisories/zones at or above this severity become hard exclusions "
             "(weight=inf). MVFR includes mountain obscuration. IFR is conservative.",
    )

    st.header("✈️ Airspace")
    operating_alt_ft = st.number_input(
        "Operating altitude (ft AGL)",
        min_value=50, max_value=1200, value=400, step=50,
        help="Used to filter Class E airspace polygons for display. "
             "Only surface-level Class E (floor ≤ this altitude) is shown as "
             "controlled. Set to 350 for this mission. "
             "⚠️ Rebuild graph with `python build_graph.py --airspace "
             "--operating-alt N` to apply to routing.",
    )
    echo_penalty = st.slider(
        "Class E penalty (min equiv.)", 0, 2000, 500, 50,
        help="Added to edge cost for Class E airspace. "
             "Hard exclusions (D/C/B/TFR/WX) cannot be overridden.",
    )
    show_airspace = st.checkbox("Show exclusion zones", value=True)

    st.subheader("💉 Inject TFR")
    with st.expander("Define synthetic TFR"):
        st.caption("Edit `data/tfrs.json` to add permanent TFRs, or inject one below.")
        tfr_name = st.text_input("Name", "Ad hoc TFR")
        tfr_coords = st.text_area(
            "Polygon vertices (lat,lon — one per line)",
            value="38.80,-78.70\n38.80,-78.20\n38.40,-78.20\n38.40,-78.70",
            height=100,
        )
        tfr_start = st.text_input("Start UTC (ISO)", "2026-05-15T14:00:00Z")
        tfr_end   = st.text_input("End UTC (ISO)",   "2026-05-15T20:00:00Z")
        if st.button("💾 Save to data/tfrs.json"):
            try:
                poly = []
                for line in tfr_coords.strip().splitlines():
                    lat, lon = line.split(",")
                    poly.append([float(lat), float(lon)])
                entry = {
                    "name":      tfr_name,
                    "polygon":   poly,
                    "start_utc": tfr_start,
                    "end_utc":   tfr_end,
                }
                p = Path("data/tfrs.json")
                existing = json.loads(p.read_text()) if p.exists() else []
                existing.append(entry)
                p.write_text(json.dumps(existing, indent=2))
                st.success("Saved. Active immediately — no graph rebuild needed.")
            except Exception as exc:
                st.error(f"Parse error: {exc}")

    st.subheader("🌩️ Inject Weather Zone")
    with st.expander("Define manual weather exclusion zone"):
        st.caption(
            "Saved to `data/weather_zones.json`. Active immediately — no rebuild needed. "
            "G-AIRMET advisories are also converted to zones automatically if the "
            "checkbox above is enabled."
        )
        wz_name     = st.text_input("Name", "Manual Wx Zone", key="wz_name")
        wz_severity = st.selectbox("Severity", ["IFR", "MVFR", "LIFR"], index=0)
        wz_hazard   = st.text_input("Hazard (informational)", "IFR", key="wz_hazard")
        wz_coords   = st.text_area(
            "Polygon vertices (lat,lon — one per line)",
            value="38.50,-79.00\n38.50,-78.50\n38.00,-78.50\n38.00,-79.00",
            height=100,
            key="wz_coords",
        )
        wz_start = st.text_input("Start UTC (ISO)", "2026-05-15T12:00:00Z", key="wz_start")
        wz_end   = st.text_input("End UTC (ISO)",   "2026-05-15T20:00:00Z", key="wz_end")
        if st.button("💾 Save to data/weather_zones.json"):
            try:
                poly = []
                for line in wz_coords.strip().splitlines():
                    lat, lon = line.split(",")
                    poly.append([float(lat), float(lon)])
                entry = {
                    "name":      wz_name,
                    "polygon":   poly,
                    "severity":  wz_severity,
                    "hazard":    wz_hazard,
                    "start_utc": wz_start,
                    "end_utc":   wz_end,
                }
                p = Path("data/weather_zones.json")
                existing = json.loads(p.read_text()) if p.exists() else []
                existing.append(entry)
                p.write_text(json.dumps(existing, indent=2))
                st.success("Saved. Active immediately — no graph rebuild needed.")
            except Exception as exc:
                st.error(f"Parse error: {exc}")

    st.header("🔍 Display")
    show_nodes = st.checkbox("Show graph nodes", value=False)

    find_btn = st.button("▶  Find Routes", type="primary", use_container_width=True)

# ── Load data ─────────────────────────────────────────────────────────────────

_fix_coords = [(f.lat, f.lon) for f in graph_data.fixes.values()
               if f.lat is not None and f.lon is not None]
_map_centre = (
    [sum(c[0] for c in _fix_coords) / len(_fix_coords),
     sum(c[1] for c in _fix_coords) / len(_fix_coords)]
    if _fix_coords else [38.0, -79.0]
)

_graph_name = pkl_choice.stem.replace("graph_", "")
_tile_dir   = Path(__file__).parent / "static" / "tiles" / _graph_name
_tile_url   = None
if _tile_dir.exists():
    _port     = st.get_option("server.port") or 8501
    _tile_url = (
        f"http://localhost:{_port}/app/static/tiles"
        f"/{_graph_name}/{{z}}/{{x}}/{{y}}.png"
    )
else:
    st.info(
        f"🗺️ No tiles found for **{_graph_name}**. "
        f"Run `python build_graph.py --tiles` to generate the network layer."
    )

_airways_in_graph = {
    v.upper()
    for v in set(data.get("airway", "") for _, _, data in G.edges(data=True))
}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", graph_data.source)
c2.metric("Nodes",  f"{G.number_of_nodes():,}")
c3.metric("Edges",  f"{G.number_of_edges()//2:,}")
has_elev = sum(1 for f in graph_data.fixes.values() if f.elevation_m is not None)
c4.metric("Elevation coverage", f"{has_elev/max(len(graph_data.fixes),1)*100:.0f}%")


# ── G-AIRMET fetch (cached 5 min) ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_gairmets(cache_path: str = "data/gairmets.cache.xml.gz") -> list[dict]:
    """
    Load G-AIRMETs from a local cache file.

    Download manually from:
        https://aviationweather.gov/data/cache/gairmets.cache.xml.gz
    Save to data/gairmets.cache.xml.gz — updated every minute by AWC.
    The app reloads it every 5 minutes (ttl=300).

    Falls back to empty list if file not present (no crash, just no overlay).
    """
    import gzip
    import xml.etree.ElementTree as ET

    p = Path(cache_path)
    if not p.exists():
        return []

    def _parse_xml_dt(adv_el, *tag_names):
        for tag in tag_names:
            txt = (adv_el.findtext(tag) or "").strip()
            if txt:
                try:
                    return datetime.fromisoformat(txt.replace("Z", "+00:00"))
                except ValueError:
                    pass
        return None

    try:
        with gzip.open(p, "rb") as f:
            root = ET.fromstring(f.read())

        advisories = []
        for adv in root.findall(".//GAIRMET"):
            hazard = adv.findtext("hazard", "").upper().strip()
            if hazard not in ("IFR", "MTN OBSCN", "MTN", "MTN_OBSCN"):
                continue

            coords_el = adv.find(".//coordinates")
            if coords_el is None or not coords_el.text:
                continue
            pts = coords_el.text.strip().split()
            if len(pts) < 6:
                continue
            try:
                ring = [[float(pts[i]), float(pts[i+1])]
                        for i in range(0, len(pts)-1, 2)]
            except (ValueError, IndexError):
                continue

            valid_from = _parse_xml_dt(
                adv, "valid", "validTime", "startTime", "starttime", "issue_time"
            )
            valid_to = _parse_xml_dt(
                adv, "expire", "endTime", "endtime", "expires", "validEnd", "end_time"
            )

            advisories.append({
                "hazard":     hazard,
                "geom":       {"type": "Polygon", "coordinates": [ring]},
                "valid_from": valid_from,
                "valid_to":   valid_to,
            })

        return advisories

    except Exception as exc:
        st.warning(f"G-AIRMET parse error: {exc}")
        return []


def _gairmets_to_weather_zones(advisories: list[dict]) -> list[WeatherZone]:
    """Convert G-AIRMET advisory dicts to WeatherZone objects."""
    _HAZARD_SEVERITY = {
        "IFR":       "IFR",
        "MTN OBSCN": "MVFR",
        "MTN":       "MVFR",
        "MTN_OBSCN": "MVFR",
    }
    zones = []
    for i, adv in enumerate(advisories):
        hazard   = (adv.get("hazard") or "").upper().strip()
        severity = _HAZARD_SEVERITY.get(hazard, "MVFR")

        geom   = adv.get("geom") or adv.get("geometry") or {}
        coords = geom.get("coordinates", [[]])
        ring   = coords[0] if coords else []
        if len(ring) < 3:
            continue

        polygon = [(pt[1], pt[0]) for pt in ring if len(pt) >= 2]
        if len(polygon) < 3:
            continue

        zones.append(WeatherZone(
            name=f"G-AIRMET {hazard} #{i + 1}",
            polygon=polygon,
            severity=severity,
            hazard=hazard,
            start_utc=adv.get("valid_from"),
            end_utc=adv.get("valid_to"),
        ))
    return zones


gairmets = fetch_gairmets("data/gairmets.cache.xml.gz") if show_gairmet else []


# ── Airspace exclusion zones (display + TFR routing) ─────────────────────────

@st.cache_data(show_spinner=False)
def get_airspace_exclusion(tfr_mtime: float, operating_agl_ft: float) -> AirspaceExclusion:
    """Cache keyed on TFR file mtime + operating altitude."""
    return load_airspace(operating_agl_ft=operating_agl_ft)


_tfr_path  = Path("data/tfrs.json")
_tfr_mtime = _tfr_path.stat().st_mtime if _tfr_path.exists() else 0.0
airspace_ex = (
    get_airspace_exclusion(_tfr_mtime, float(operating_alt_ft))
    if show_airspace else None
)


# ── Manual weather zones ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_manual_weather_zones(wz_mtime: float) -> list[WeatherZone]:
    """Cache keyed on weather_zones.json mtime."""
    return load_weather_zones()


_wz_path  = Path("data/weather_zones.json")
_wz_mtime = _wz_path.stat().st_mtime if _wz_path.exists() else 0.0
_manual_wx_zones = get_manual_weather_zones(_wz_mtime)

_gairmet_wx_zones: list[WeatherZone] = (
    _gairmets_to_weather_zones(gairmets) if apply_gairmet_routing and gairmets else []
)
_all_weather_zones: list[WeatherZone] = _gairmet_wx_zones + _manual_wx_zones


# ── Re-weight graph + route ───────────────────────────────────────────────────

_exempt = []
for _name in (origin_input, dest_input):
    try:
        _fix = graph_data.fixes.get(
            resolve_endpoint(_name, graph_data) if _name else None
        )
        if _fix:
            _exempt.append((_fix.lat, _fix.lon))
    except Exception:
        pass

apply_weights(
    G,
    time_weight=time_weight,
    elev_weight=elev_weight,
    noise_weight=noise_weight,
    wx_filter=wx_filter,
    echo_penalty=float(echo_penalty),
    exempt_coords=_exempt or None,
    exempt_radius_nm=10.0,   # tightened from 25 nm — prevents PHF/LFI/FAF pass-through
    current_tfrs=airspace_ex.tfrs if airspace_ex else None,
    current_weather_zones=_all_weather_zones or None,
    wx_zone_severity_threshold=wx_zone_threshold,
)

routes    = None
status    = None
origin_id = None
dest_id   = None

if find_btn:
    try:
        origin_id = resolve_endpoint(origin_input, graph_data)
        dest_id   = resolve_endpoint(dest_input,   graph_data)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    dep = datetime.now(tz=timezone.utc)
    with st.spinner("Pathfinding…"):
        rs = find_routes(G, origin_id, dest_id, graph_data, dep, cruise_kts, k_routes)
    routes = rs.routes
    status = (f"✅ {len(routes)} route(s)  ·  {origin_id!r} → {dest_id!r}"
              if routes else
              "⚠️ No routes found. The graph may be disconnected or all paths excluded.")

if status:
    st.info(status)


# ── Elevation profile helpers ─────────────────────────────────────────────────

def _fetch_elevation_profiles(routes_list, gdata) -> list[dict]:
    """
    Fetch SRTM 30m terrain elevation along every route in one batched pass.

    Collects all interpolated sample points across all routes, batches them
    into groups of 100 (Open-Topo-Data API hard limit), then reassembles into
    per-route arrays.  Typical k=3 route set: 2-4 API calls, ~3-5 seconds.

    Returns list of dicts, one per route:
        dist_nm         : list[float]  cumulative distance axis (nm)
        terrain_ft_msl  : list[float]  terrain elevation in ft MSL
        drone_ft_msl    : list[float]  terrain + operating_agl_ft (drone MSL track)
        waypoint_dist_nm: list[float]  cumulative distance at each graph node
    """
    _op_ft = float(operating_alt_ft)

    # ── Collect all sample points ─────────────────────────────────────────
    all_pts: list[tuple] = []    # (lat, lon, route_idx, seg_idx, pt_idx)
    meta: list[dict]     = []    # per-route geometry info

    for ri, route in enumerate(routes_list):
        seg_info   = []           # (cum_dist_at_start, seg_dist_nm)
        wp_dists   = [0.0]
        cum        = 0.0

        for si, seg in enumerate(route.segments):
            fa = gdata.fixes.get(seg.from_ident)
            fb = gdata.fixes.get(seg.to_ident)
            if fa is None or fb is None:
                seg_info.append((cum, 0.0, None, None))
                continue

            d = seg.distance_nm
            seg_info.append((cum, d, fa, fb))

            for pi in range(_ELEV_SAMPLES):
                t = pi / (_ELEV_SAMPLES - 1)
                lat = fa.lat + t * (fb.lat - fa.lat)
                lon = fa.lon + t * (fb.lon - fa.lon)
                all_pts.append((lat, lon, ri, si, pi))

            cum += d
            wp_dists.append(cum)

        meta.append({"seg_info": seg_info, "wp_dists": wp_dists, "total_nm": cum})

    if not all_pts:
        return [{"dist_nm": [], "terrain_ft_msl": [], "drone_ft_msl": [],
                 "waypoint_dist_nm": []} for _ in routes_list]

    # ── Batched API fetch ─────────────────────────────────────────────────
    elevations = [0.0] * len(all_pts)

    for i in range(0, len(all_pts), _ELEV_BATCH):
        batch    = all_pts[i: i + _ELEV_BATCH]
        loc_str  = "|".join(f"{p[0]:.6f},{p[1]:.6f}" for p in batch)
        try:
            resp = requests.post(
                _ELEV_API_URL,
                json={"locations": loc_str},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for j, result in enumerate(results):
                elevations[i + j] = float(result.get("elevation") or 0.0)
        except Exception:
            pass   # leave zeros — don't crash, profile just shows flat
        if i + _ELEV_BATCH < len(all_pts):
            time.sleep(_ELEV_DELAY)

    # ── Build lookup and reassemble ───────────────────────────────────────
    elev_map: dict[tuple, float] = {
        (pt[2], pt[3], pt[4]): elevations[k]
        for k, pt in enumerate(all_pts)
    }

    profiles = []
    for ri, route in enumerate(routes_list):
        m       = meta[ri]
        dist_arr: list[float] = []
        terr_arr: list[float] = []

        for si, (cum_d, seg_d, fa, fb) in enumerate(m["seg_info"]):
            if fa is None:
                continue
            for pi in range(_ELEV_SAMPLES):
                t  = pi / (_ELEV_SAMPLES - 1)
                d  = cum_d + t * seg_d
                em = elev_map.get((ri, si, pi), 0.0)
                ef = em * 3.28084          # metres → feet
                # Skip duplicate boundary points between segments
                if dist_arr and abs(d - dist_arr[-1]) < 1e-6:
                    continue
                dist_arr.append(round(d, 3))
                terr_arr.append(round(ef, 1))

        drone_arr = [t + _op_ft for t in terr_arr]

        profiles.append({
            "dist_nm":          dist_arr,
            "terrain_ft_msl":   terr_arr,
            "drone_ft_msl":     drone_arr,
            "waypoint_dist_nm": m["wp_dists"],
        })

    return profiles


# ── Map ───────────────────────────────────────────────────────────────────────

def make_map(
    graph_data, G, routes, gairmets, airspace_ex,
    show_nodes, map_centre, tile_url, airways_in_graph,
    weather_zones=None,
) -> folium.Map:
    m = folium.Map(location=map_centre, zoom_start=7, tiles="CartoDB positron")

    # ── Graph network tile layer ──────────────────────────────────────────
    if tile_url:
        folium.TileLayer(
            tiles=tile_url,
            attr="Trailblazer graph tiles",
            name="Graph network",
            overlay=True,
            control=True,
            opacity=1.0,
            max_native_zoom=11,
            max_zoom=18,
        ).add_to(m)

    # ── Airspace exclusion zones ──────────────────────────────────────────
    if airspace_ex:
        airspace_grp = folium.FeatureGroup("Airspace zones", show=True)
        if getattr(airspace_ex, "_use_shapefile", False) and airspace_ex._gdf is not None:
            import json as _json
            gdf = airspace_ex._gdf

            if _fix_coords:
                _lats = [c[0] for c in _fix_coords]
                _lons = [c[1] for c in _fix_coords]
                _buf  = 1.0
                _bbox = (
                    min(_lons) - _buf, min(_lats) - _buf,
                    max(_lons) + _buf, max(_lats) + _buf,
                )
                try:
                    from shapely.geometry import box as _box
                    gdf = gdf[gdf.geometry.intersects(_box(*_bbox))]
                except Exception:
                    pass

            for cls, color in AIRSPACE_COLORS.items():
                if not color or cls == "WX":
                    continue
                subset = gdf[gdf["_class"] == cls]
                if subset.empty:
                    continue
                try:
                    _has_ident = "_ident" in gdf.columns
                    cols    = (["_ident", "_class", "geometry"]
                               if _has_ident else ["_class", "geometry"])
                    geojson = _json.loads(subset[cols].to_json())
                    tooltip = (
                        folium.GeoJsonTooltip(
                            fields=["_ident", "_class"],
                            aliases=["Airport", "Class"],
                            localize=True,
                        ) if _has_ident else
                        folium.GeoJsonTooltip(fields=["_class"], aliases=["Class"])
                    )
                    folium.GeoJson(
                        geojson,
                        style_function=lambda f, c=color: {
                            "color": c, "weight": 1.5, "opacity": 0.7,
                            "fillColor": c, "fillOpacity": 0.10,
                        },
                        tooltip=tooltip,
                        name=f"Class {cls}",
                    ).add_to(airspace_grp)
                except Exception:
                    pass
        else:
            for zone in airspace_ex.zones:
                color = AIRSPACE_COLORS.get(zone["airspace"], IBM["gray"])
                if color is None:
                    continue
                folium.Circle(
                    [zone["lat"], zone["lon"]],
                    radius=zone["radius_nm"] * 1852,
                    color=color, fill=True, fill_opacity=0.12,
                    weight=1.5, opacity=0.6,
                    tooltip=f'{zone["ident"]} · Class {zone["airspace"]}',
                ).add_to(airspace_grp)

        # TFR polygons — active only
        _now_utc = datetime.now(timezone.utc)
        for tfr in airspace_ex.tfrs:
            if tfr.end_utc and _now_utc > tfr.end_utc:
                continue
            if tfr.start_utc and _now_utc < tfr.start_utc:
                continue
            folium.Polygon(
                locations=tfr.polygon,
                color=IBM["red"], fill=True, fill_opacity=0.20, weight=2,
                tooltip=f"TFR: {tfr.name}",
            ).add_to(airspace_grp)

        # DC SFRA
        from trailblazer.airspace.exclusion import _KDCA_LAT, _KDCA_LON, _SFRA_RADIUS_NM
        folium.Circle(
            [_KDCA_LAT, _KDCA_LON],
            radius=_SFRA_RADIUS_NM * 1852,
            color=IBM["red"], fill=True, fill_opacity=0.10,
            weight=2, dash_array="6",
            tooltip="DC SFRA",
        ).add_to(airspace_grp)
        airspace_grp.add_to(m)

    # ── Weather exclusion zones ───────────────────────────────────────────
    if weather_zones:
        wx_zone_grp = folium.FeatureGroup("Weather Zones", show=True)
        _now_utc = datetime.now(timezone.utc)
        for wz in weather_zones:
            if wz.start_utc and _now_utc < wz.start_utc:
                continue
            if wz.end_utc and _now_utc > wz.end_utc:
                continue
            color = WX_COLORS.get(wz.severity, IBM["teal"]) or IBM["teal"]
            folium.Polygon(
                locations=wz.polygon,
                color=color, fill=True, fill_opacity=0.20, weight=2,
                dash_array="6 4",
                tooltip=f"Wx: {wz.name} · {wz.severity} · {wz.hazard or '—'}",
            ).add_to(wx_zone_grp)
        wx_zone_grp.add_to(m)

    # ── G-AIRMET overlay (display layer — independent of routing zones) ───
    if gairmets:
        wx_grp = folium.FeatureGroup("G-AIRMETs", show=True)
        for adv in gairmets:
            hazard = (adv.get("hazard") or "").upper()
            fc     = "IFR" if hazard == "IFR" else "MVFR"
            color  = WX_COLORS.get(fc, IBM["gray"])
            if not color:
                continue
            geom  = adv.get("geom") or adv.get("geometry") or {}
            gtype = geom.get("type", "")
            polys = []
            if gtype == "Polygon":
                polys = [geom.get("coordinates", [[]])[0]]
            elif gtype == "MultiPolygon":
                for poly in geom.get("coordinates", []):
                    polys.append(poly[0])
            for ring in polys:
                locs = [[pt[1], pt[0]] for pt in ring if len(pt) >= 2]
                if len(locs) < 3:
                    continue
                folium.Polygon(
                    locs, color=color, fill=True, fill_opacity=0.10, weight=1,
                    dash_array="2 4",
                    tooltip=f"G-AIRMET {hazard} ({fc})",
                ).add_to(wx_grp)
        wx_grp.add_to(m)

    # ── Graph nodes ───────────────────────────────────────────────────────
    if show_nodes:
        node_grp = folium.FeatureGroup("Nodes", show=True)
        node_features = []
        for fix in graph_data.fixes.values():
            if fix.ident not in G:
                continue
            elev = f"{fix.elevation_m:.0f} m" if fix.elevation_m is not None else "—"
            node_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(fix.lon, 5), round(fix.lat, 5)],
                },
                "properties": {"ident": fix.ident, "elev": elev},
            })
        folium.GeoJson(
            {"type": "FeatureCollection", "features": node_features},
            marker=folium.CircleMarker(
                radius=2, color=IBM["gray"], fill=True, fill_opacity=0.6,
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=["ident", "elev"], aliases=["Node", "Elev"],
            ),
            name="Nodes",
        ).add_to(node_grp)
        node_grp.add_to(m)

    # ── Routes ────────────────────────────────────────────────────────────
    if routes:
        route_grp = folium.FeatureGroup("Routes", show=True)
        for r in routes:
            color = ROUTE_COLORS[min(r.rank - 1, len(ROUTE_COLORS) - 1)]
            coords = [
                (graph_data.fixes[fid].lat, graph_data.fixes[fid].lon)
                for fid in r.path if fid in graph_data.fixes
            ]
            if len(coords) < 2:
                continue
            h, mn = divmod(int(r.time_min), 60)
            folium.PolyLine(
                coords, color=color, weight=6 - r.rank, opacity=0.95,
                tooltip=f"#{r.rank} · {r.distance_nm:.0f} nm · {h}h{mn:02d}m · wx:{r.worst_wx}",
            ).add_to(route_grp)
            for fid in (r.path[0], r.path[-1]):
                fix = graph_data.fixes.get(fid)
                if fix:
                    folium.CircleMarker(
                        [fix.lat, fix.lon], radius=8,
                        color=color, fill=True, fill_opacity=0.9,
                        tooltip=fix.ident,
                    ).add_to(route_grp)
        route_grp.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────
    # Network voltage encoded by lineweight (all lines same colour).
    _lw_labels = [
        ("765KV", "765 kV", 4), ("500KV", "500 kV", 3),
        ("345KV", "345 kV", 3), ("230KV", "230 kV", 2),
        ("115KV", "115 kV", 2), ("TOWER_MESH", "Cell tower mesh", 1),
    ]
    items = []
    for key, label, lw_px in _lw_labels:
        if key in airways_in_graph and tile_url:
            items.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                f'<div style="width:26px;height:{lw_px}px;background:{_NETWORK_COLOR};'
                f'border-radius:1px"></div>'
                f'<span style="font-size:11px">{label}</span></div>'
            )
    items.append('<hr style="margin:4px 0">')
    for cls, label in [("D","Class D (excl.)"),("C","Class C (excl.)"),
                        ("B","Class B (excl.)"),("TFR","TFR (excl.)")]:
        c = AIRSPACE_COLORS.get(cls, IBM["gray"])
        if c:
            items.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                f'<div style="width:26px;height:4px;background:{c};border-radius:2px;'
                f'opacity:0.7"></div>'
                f'<span style="font-size:11px">{label}</span></div>'
            )
    items.append(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<div style="width:26px;height:4px;background:{IBM["teal"]};border-radius:2px;'
        f'opacity:0.8;border:1px dashed {IBM["teal"]}"></div>'
        f'<span style="font-size:11px">Wx Zone (excl.)</span></div>'
    )
    if routes:
        items.append('<hr style="margin:4px 0">')
        for r in routes:
            c = ROUTE_COLORS[min(r.rank-1, len(ROUTE_COLORS)-1)]
            h, mn = divmod(int(r.time_min), 60)
            items.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                f'<div style="width:26px;height:4px;background:{c};border-radius:2px"></div>'
                f'<span style="font-size:11px">Route #{r.rank} · '
                f'{r.distance_nm:.0f} nm · {h}h{mn:02d}m</span></div>'
            )

    legend = (
        '<div style="position:fixed;bottom:28px;right:8px;z-index:1000;'
        'background:white;padding:10px 14px;border-radius:8px;'
        'box-shadow:0 1px 5px rgba(0,0,0,.25);font-family:sans-serif;min-width:180px">'
        '<div style="font-size:12px;font-weight:700;margin-bottom:4px">Legend</div>'
        + "".join(items) + '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


m = make_map(
    graph_data, G, routes, gairmets,
    airspace_ex if show_airspace else None,
    show_nodes, _map_centre, _tile_url, _airways_in_graph,
    weather_zones=_all_weather_zones or None,
)
st_folium(m, width="100%", height=640, returned_objects=[])

# ── Route table ───────────────────────────────────────────────────────────────

if routes:
    st.subheader("Routes")
    rows = []
    for r in routes:
        h, mn = divmod(int(r.time_min), 60)
        _e_pct = (r.class_e_time_min / r.time_min * 100) if r.time_min > 0 else 0
        rows.append({
            "Rank":         f"#{r.rank}",
            "Dist (nm)":    f"{r.distance_nm:.0f}",
            "ETE":          f"{h}h{mn:02d}m",
            "Cost":         f"{r.cost:.1f}",
            "Corridors":    ", ".join(r.unique_airways[:4]),
            "Worst Wx":     r.worst_wx,
            "Class E time": f"{r.class_e_time_min:.0f} min ({_e_pct:.0f}%)",
            "Climb (m)":    f"{r.total_climb_m:.0f}" if r.total_climb_m else "—",
            "Waypoints":    len(r.path),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Segment detail — Route #1"):
        segs = []
        for i, seg in enumerate(routes[0].segments):
            segs.append({
                "Leg":       i + 1,
                "From":      seg.from_ident,
                "To":        seg.to_ident,
                "Corridor":  seg.airway_id,
                "kV":        f"{seg.voltage_kv:.0f}" if seg.voltage_kv else "—",
                "Airspace":  G[seg.from_ident][seg.to_ident].get("airspace_class", "?"),
                "Wx":        seg.wx_category,
                "Dist (nm)": f"{seg.distance_nm:.1f}",
                "ETE (min)": f"{seg.time_min:.1f}",
                "Elev gain": f"{G[seg.from_ident][seg.to_ident].get('elev_gain_m',0):.0f} m",
            })
        st.dataframe(pd.DataFrame(segs), use_container_width=True, hide_index=True)


# ── Elevation profile ─────────────────────────────────────────────────────────
# Profiles are fetched once when Find Routes is clicked and persisted in
# session_state so they survive subsequent slider moves (which re-render the
# app but don't re-click the button).

if find_btn and routes:
    _prof_key = (origin_id, dest_id)
    if st.session_state.get("_elev_profile_key") != _prof_key:
        with st.spinner(f"Fetching terrain profile (SRTM 30m, {len(routes)} route(s))…"):
            _profiles = _fetch_elevation_profiles(routes, graph_data)
        st.session_state["_elev_profiles"]    = _profiles
        st.session_state["_elev_routes"]      = routes
        st.session_state["_elev_profile_key"] = _prof_key

_elev_profiles = st.session_state.get("_elev_profiles")
_elev_routes   = st.session_state.get("_elev_routes")

if _elev_profiles and _elev_routes:
    with st.expander("📈 Terrain profile", expanded=True):
        # Route selector — only shown when k > 1
        if len(_elev_routes) > 1:
            _r_label = st.selectbox(
                "Route",
                options=[f"#{r.rank}  ({r.distance_nm:.0f} nm  ·  {r.worst_wx})"
                         for r in _elev_routes],
                key="elev_profile_route_sel",
                label_visibility="collapsed",
            )
            _ri = int(_r_label.split("#")[1].split()[0]) - 1
        else:
            _ri = 0

        _prof = _elev_profiles[_ri] if _ri < len(_elev_profiles) else None

        if _prof and _prof["dist_nm"]:
            try:
                import altair as alt

                _op_ft   = float(operating_alt_ft)
                _dist    = _prof["dist_nm"]
                _terr    = _prof["terrain_ft_msl"]
                _drone   = _prof["drone_ft_msl"]
                _wp_dist = _prof["waypoint_dist_nm"]

                _df = pd.DataFrame({
                    "Distance (nm)":        _dist,
                    "Terrain (ft MSL)":     _terr,
                    "Drone track (ft MSL)": _drone,   # terrain + operating AGL
                })

                # Terrain filled area
                _base = alt.Chart(_df).encode(
                    x=alt.X("Distance (nm):Q",
                            axis=alt.Axis(title="Distance (nm)", grid=False)),
                )
                _terrain_area = _base.mark_area(
                    color=IBM["teal"], opacity=0.30,
                    line={"color": IBM["teal"], "strokeWidth": 1.2},
                ).encode(
                    y=alt.Y("Terrain (ft MSL):Q",
                            scale=alt.Scale(zero=True),
                            axis=alt.Axis(title="Altitude (ft MSL)")),
                    tooltip=[
                        alt.Tooltip("Distance (nm):Q", format=".1f"),
                        alt.Tooltip("Terrain (ft MSL):Q", format=".0f"),
                    ],
                )

                # Drone track dashed line
                _drone_line = _base.mark_line(
                    color=IBM["orange"], strokeDash=[5, 3], strokeWidth=1.5,
                ).encode(
                    y=alt.Y("Drone track (ft MSL):Q"),
                    tooltip=[
                        alt.Tooltip("Distance (nm):Q", format=".1f"),
                        alt.Tooltip("Drone track (ft MSL):Q", format=".0f",
                                    title=f"Drone ({_op_ft:.0f} ft AGL)"),
                    ],
                )

                # Waypoint tick marks
                _layers = [_terrain_area, _drone_line]
                if _wp_dist:
                    _wp_df = pd.DataFrame({"Distance (nm)": _wp_dist})
                    _wp_rules = (
                        alt.Chart(_wp_df)
                        .mark_rule(color=IBM["gray"], strokeWidth=0.6, opacity=0.5)
                        .encode(x="Distance (nm):Q")
                    )
                    _layers.append(_wp_rules)

                _chart = (
                    alt.layer(*_layers)
                    .properties(height=200)
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelFontSize=11, titleFontSize=11)
                )
                st.altair_chart(_chart, use_container_width=True)

            except ImportError:
                # Altair not available — fall back to st.line_chart
                _df_simple = pd.DataFrame({
                    "Terrain (ft MSL)":     _prof["terrain_ft_msl"],
                    "Drone track (ft MSL)": _prof["drone_ft_msl"],
                }, index=_prof["dist_nm"])
                st.line_chart(_df_simple)

            # Summary stats
            _max_t   = max(_prof["terrain_ft_msl"]) if _prof["terrain_ft_msl"] else 0
            _max_d   = max(_prof["drone_ft_msl"])   if _prof["drone_ft_msl"]   else 0
            _min_clr = min(
                d - t for d, t in zip(_prof["drone_ft_msl"], _prof["terrain_ft_msl"])
            ) if _prof["terrain_ft_msl"] else float(operating_alt_ft)

            st.caption(
                f"Peak terrain: **{_max_t:.0f} ft MSL** · "
                f"Peak drone altitude: **{_max_d:.0f} ft MSL** · "
                f"Min AGL clearance: **{_min_clr:.0f} ft** · "
                f"Operating AGL: {operating_alt_ft:.0f} ft"
            )
        else:
            st.caption("Terrain profile unavailable — Open-Topo-Data API unreachable.")


# ── Weather zone status summary ───────────────────────────────────────────────

_active_display_zones = [
    wz for wz in _all_weather_zones
    if not (wz.start_utc and datetime.now(timezone.utc) < wz.start_utc)
    and not (wz.end_utc and datetime.now(timezone.utc) > wz.end_utc)
]
if _active_display_zones:
    with st.expander(f"🌩️ Active weather zones ({len(_active_display_zones)})"):
        wz_rows = []
        _RANK = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}
        _threshold_rank = _RANK.get(wx_zone_threshold.upper(), 2)
        for wz in _active_display_zones:
            zone_rank = _RANK.get(wz.severity, 2)
            routing_status = (
                "Hard exclusion ✖" if zone_rank >= _threshold_rank else
                "Display only (below threshold)"
            )
            wz_rows.append({
                "Name":     wz.name,
                "Severity": wz.severity,
                "Hazard":   wz.hazard or "—",
                "Routing":  routing_status,
                "Source":   "G-AIRMET" if wz.name.startswith("G-AIRMET") else "Manual",
            })
        st.dataframe(pd.DataFrame(wz_rows), use_container_width=True, hide_index=True)

with st.expander("ℹ️ How sliders work"):
    st.markdown("""
**Sliders recompute edge weights in-place — no graph rebuild needed.**

Each edge stores component costs at build time:
`time_min`, `elev_gain_m`, `noise_score`, `wx_rank`, `airspace_class`.

When you move a slider, `apply_weights(G, λ)` recomputes
`weight = λ_t·time + λ_e·elev_gain + λ_n·noise + airspace_penalty`
across all edges in milliseconds, then pathfinding re-runs on the new weights.

**Noise λ_n** is wired but scores 0 until LandScan USA data is loaded (Phase 4).
**Altitude λ_e** requires elevations fetched with `python build_graph.py --elevation`.
**Airspace** classification requires `--airspace` flag at build time.

**Weather zones** are live polygon exclusions applied on every re-weight pass —
no rebuild needed. G-AIRMET advisories are converted to zones automatically.
The severity threshold slider controls which zones become hard exclusions (weight=inf).

**Operating altitude** affects the Class E display layer immediately.
To apply it to routing, rebuild: `python build_graph.py --airspace --operating-alt 350`.

**Terrain profile** is fetched from Open-Topo-Data SRTM 30m after each "Find Routes"
click and persists across slider moves. Orange dashed line = drone MSL track
(terrain + operating AGL). Teal fill = terrain MSL. Vertical ticks = graph waypoints.
""")