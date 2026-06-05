"""
trailblazer/routing/pathfinder.py — Graph construction and k-shortest path routing

Builds a NetworkX DiGraph from any GraphData source (transmission lines,
NASR airways, or cell tower mesh) filtered against a WeatherProvider, then
finds the k best routes using Yen's algorithm.

Edge weather is sampled at the estimated arrival time at each segment
midpoint — not at departure time.  A 3-hour route through a moving weather
system therefore gets time-accurate wx at each waypoint, not a T=0 snapshot.

Objective function
──────────────────
    cost(edge) = λ1 · time_min
               + λ2 · social_impact     ← Phase 4; stub returns 0.0 until then
               + λ3 · traffic_impact    ← future; stub returns 0.0
               - λ4 · C2_margin        ← cell graph only; future

Hard filters (edge excluded entirely — not penalised):
    wx_category > wx_threshold
    airport_buffer_intersect           ← Phase 2
    TFR_intersect                      ← Phase 2
    WeatherZone_intersect              ← Phase 3 (live, same mechanism as TFR)

Voltage discount on social_impact (Phase 4):
    765 kV → 0.10×    500 kV → 0.20×    345 kV → 0.40×
    230 kV → 0.65×    115 kV → 0.85×    off-ROW → 1.00×
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import networkx as nx

from ..types import Fix, AirwaySegment, GraphData
from ..weather.provider import WeatherProvider, FC_RANK, worst_category
from ..scoring.social import social_impact, voltage_multiplier


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class RouteSegment:
    """Weather and geometry data for a single graph edge."""
    from_ident:  str
    to_ident:    str
    airway_id:   str
    distance_nm: float
    time_min:    float
    wx_category: str
    voltage_kv:  float = 0.0
    # Phase 4 slots:
    # social_score: float = 0.0
    # noise_db_pop: float = 0.0


@dataclass
class Route:
    """A complete origin→destination route."""
    rank:        int
    segments:    list[RouteSegment]
    distance_nm: float
    time_min:    float
    worst_wx:    str
    cost:        float

    class_e_time_min: float = 0.0   # minutes in Class E airspace (incursion metric)
    total_climb_m:    float = 0.0   # total ascent in metres (climb metric)
    total_pop_affected: float = 0.0 # sum of pop_sum along route corridor (raw, not normalised)

    @property
    def path(self) -> list[str]:
        if not self.segments:
            return []
        return [self.segments[0].from_ident] + [s.to_ident for s in self.segments]

    @property
    def airways(self) -> list[str]:
        return [s.airway_id for s in self.segments]

    @property
    def unique_airways(self) -> list[str]:
        seen, result = set(), []
        for a in self.airways:
            if a not in seen:
                seen.add(a)
                result.append(a)
        return result


@dataclass
class RouteSet:
    """Output of a single pathfinder run."""
    origin:         str
    destination:    str
    departure_time: datetime
    cruise_kts:     float
    wx_filter:      str
    graph_source:   str          # "transmission" | "nasr" | "cell"
    routes:         list[Route]
    fixes:          dict[str, Fix]


# ── Objective function ────────────────────────────────────────────────────────

def edge_cost(
    dist_nm:        float,
    time_min:       float,
    wx_category:    str,
    voltage_kv:     float = 0.0,
    lat1: float = 0.0, lon1: float = 0.0,
    lat2: float = 0.0, lon2: float = 0.0,
    elev1_m: float | None = None,
    elev2_m: float | None = None,
    cruise_kts: float = 120.0,
    departure_time: Optional[datetime] = None,
    # Objective function weights
    wx_penalty_weight:  float = 0.0,
    noise_weight:       float = 0.0,   # λ2 — social impact slider in Streamlit
    elev_weight:        float = 0.0,   # λ_e — altitude penalty (potential energy proxy)
    traffic_weight:     float = 0.0,   # λ3 — future
    c2_margin_weight:   float = 0.0,   # λ4 — cell graph only
) -> float:
    """
    Compute edge routing cost (in units of equivalent flight-minutes).

    Phase 1–3: pure ETE.  wx hard-filters edges; noise_weight=0 until Phase 4.

    Phase 4: set noise_weight > 0 and pass a loaded pop_raster to unlock
    the social impact term.  The Streamlit slider drives noise_weight.

    Voltage discount: transmission ROW reduces social_impact by a fixed
    multiplier (see scoring/social.py).  The discount is applied inside
    social_impact() — this function just passes voltage_kv through.

    Soft vs hard wx: to switch from hard exclusion to soft penalty, set
    wx_penalty_weight > 0 and wx_filter="LIFR" in build_graph().  This
    allows routing through MVFR at a cost rather than returning no-route.
    """
    wx_pen = FC_RANK.get(wx_category.upper(), 4) * wx_penalty_weight

    noise_score = 0.0
    if noise_weight > 0.0:
        noise_score = social_impact(
            lat1, lon1, lat2, lon2,
            cruise_kts=cruise_kts,
            voltage_kv=voltage_kv,
            departure_time=departure_time,
        )

    elev_score = 0.0
    if elev_weight > 0.0 and elev1_m is not None and elev2_m is not None:
        elev_score = max(0.0, elev2_m - elev1_m)

    traffic_score = 0.0
    c2_score = 0.0

    return (
        time_min
        + wx_pen
        + noise_weight   * noise_score
        + elev_weight    * elev_score
        + traffic_weight * traffic_score
        - c2_margin_weight * c2_score
    )


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    graph_data: GraphData,
    wx_provider: WeatherProvider,
    departure_time: datetime,
    cruise_kts: float = 120.0,
    wx_filter: str = "MVFR",
    wx_penalty_weight: float = 0.0,
    noise_weight: float = 0.0,
    elev_weight: float = 0.0,
    airspace_filter=None,   # AirspaceExclusion instance, or None
) -> nx.DiGraph:
    """
    Construct a weighted directed graph from any GraphData source.

    Edge attributes
    ───────────────
    airway       : airway ID / voltage label string
    distance_nm  : great-circle distance
    time_min     : ETE at cruise_kts
    wx_category  : flight category at segment midpoint
    wx_rank      : integer rank (0=VFR … 3=LIFR)
    voltage_kv   : nominal voltage (0 for off-ROW)
    noise_score  : normalised population impact [0,1]; 0.0 until --landscan used
    pop_sum      : raw population buffer sum; set by compute_population_scores()
    weight       : routing cost for Yen's algorithm

    Segments failing the wx_filter are excluded entirely.
    Set wx_penalty_weight > 0 and wx_filter="LIFR" for soft-penalty mode.
    """
    G = nx.DiGraph()
    G.graph.update({
        "wx_filter":    wx_filter,
        "cruise_kts":   cruise_kts,
        "graph_source": graph_data.source,
    })

    min_rank     = FC_RANK.get(wx_filter.upper(), 1)
    excluded_wx  = 0
    missing_fix  = 0
    added        = 0

    dep_utc = (
        departure_time.astimezone(timezone.utc)
        if departure_time.tzinfo
        else departure_time.replace(tzinfo=timezone.utc)
    )

    for seg in graph_data.segments:
        a_id, b_id = seg.from_ident, seg.to_ident
        fix_a = graph_data.fixes.get(a_id)
        fix_b = graph_data.fixes.get(b_id)

        if fix_a is None or fix_b is None:
            missing_fix += 1
            continue

        dist_nm  = haversine_nm(fix_a.lat, fix_a.lon, fix_b.lat, fix_b.lon)
        time_min = dist_nm / cruise_kts * 60.0

        mid_lat  = (fix_a.lat + fix_b.lat) / 2
        mid_lon  = (fix_a.lon + fix_b.lon) / 2
        t_mid    = dep_utc + timedelta(minutes=time_min / 2)

        wx_cat  = wx_provider.flight_category(mid_lat, mid_lon, t_mid)
        wx_rank = FC_RANK.get(wx_cat.upper(), 4)

        if wx_rank > min_rank:
            excluded_wx += 1
            continue

        cost = edge_cost(
            dist_nm=dist_nm,
            time_min=time_min,
            wx_category=wx_cat,
            voltage_kv=seg.voltage_kv,
            lat1=fix_a.lat, lon1=fix_a.lon,
            lat2=fix_b.lat, lon2=fix_b.lon,
            elev1_m=fix_a.elevation_m,
            elev2_m=fix_b.elevation_m,
            cruise_kts=cruise_kts,
            departure_time=dep_utc,
            wx_penalty_weight=wx_penalty_weight,
            noise_weight=noise_weight,
            elev_weight=elev_weight,
        )

        # Airspace classification: full segment intersection, not just midpoint.
        airspace_class = "G"
        if airspace_filter is not None:
            airspace_class = airspace_filter.classify_segment(
                fix_a.lat, fix_a.lon, fix_b.lat, fix_b.lon, t_mid
            )

        elev_gain_m = 0.0
        if fix_a.elevation_m is not None and fix_b.elevation_m is not None:
            elev_gain_m = max(0.0, fix_b.elevation_m - fix_a.elevation_m)

        attrs = dict(
            airway=seg.airway_id,
            distance_nm=round(dist_nm, 2),
            time_min=round(time_min, 2),
            wx_category=wx_cat,
            wx_rank=wx_rank,
            voltage_kv=seg.voltage_kv,
            airspace_class=airspace_class,
            elev_gain_m=round(elev_gain_m, 1),
            noise_score=0.0,    # Phase 4: overwritten by apply_weights() when pop_sum is set
            pop_sum=0.0,        # raw population buffer sum; set by compute_population_scores()
            weight=cost,
        )

        G.add_node(a_id, lat=fix_a.lat, lon=fix_a.lon, name=fix_a.name, state=fix_a.state)
        G.add_node(b_id, lat=fix_b.lat, lon=fix_b.lon, name=fix_b.name, state=fix_b.state)
        G.add_edge(a_id, b_id, **attrs)
        G.add_edge(b_id, a_id, **attrs)
        added += 1

    print(
        f"  [Graph] {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} directed edges "
        f"(+{added} segs, -{excluded_wx} wx-filtered, -{missing_fix} missing fix)"
    )
    return G


# ── Routing ───────────────────────────────────────────────────────────────────

def find_routes(
    G: nx.DiGraph,
    origin: str,
    destination: str,
    graph_data: GraphData,
    departure_time: datetime,
    cruise_kts: float = 120.0,
    k: int = 3,
) -> RouteSet:
    """
    Find k best routes using Yen's algorithm (nx.shortest_simple_paths).

    Paths are yielded in ascending cost order.  Returns a RouteSet with the
    top k routes and all fix metadata needed by downstream export modules.
    """
    wx_filter    = G.graph.get("wx_filter",    "MVFR")
    graph_source = G.graph.get("graph_source", "unknown")
    cruise       = G.graph.get("cruise_kts",   cruise_kts)

    for fix_id in (origin, destination):
        if fix_id not in G:
            raise ValueError(
                f"Fix '{fix_id}' not in graph. "
                f"Verify the ident exists in the source data."
            )

    try:
        path_gen = nx.shortest_simple_paths(G, origin, destination, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        print(f"  [Router] No path found: {exc}")
        return RouteSet(
            origin=origin, destination=destination,
            departure_time=departure_time, cruise_kts=cruise_kts,
            wx_filter=wx_filter, graph_source=graph_source,
            routes=[], fixes=graph_data.fixes,
        )

    routes: list[Route] = []

    for path in path_gen:
        if len(routes) >= k:
            break

        segments: list[RouteSegment] = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = G[a][b]
            segments.append(RouteSegment(
                from_ident=a,
                to_ident=b,
                airway_id=e["airway"],
                distance_nm=e["distance_nm"],
                time_min=e["time_min"],
                wx_category=e["wx_category"],
                voltage_kv=e.get("voltage_kv", 0.0),
            ))

        total_dist   = sum(s.distance_nm for s in segments)
        total_time   = sum(s.time_min    for s in segments)
        worst_wx     = worst_category([s.wx_category for s in segments])
        cost         = sum(G[s.from_ident][s.to_ident]["weight"] for s in segments)
        class_e_time = sum(
            G[s.from_ident][s.to_ident].get("time_min", 0.0)
            for s in segments
            if G[s.from_ident][s.to_ident].get("airspace_class") == "E"
        )
        total_climb  = sum(
            G[s.from_ident][s.to_ident].get("elev_gain_m", 0.0)
            for s in segments
        )
        total_pop    = sum(
            G[s.from_ident][s.to_ident].get("pop_sum", 0.0)
            for s in segments
        )

        routes.append(Route(
            rank=len(routes) + 1,
            segments=segments,
            distance_nm=round(total_dist, 1),
            time_min=round(total_time, 1),
            worst_wx=worst_wx,
            cost=round(cost, 2),
            class_e_time_min=round(class_e_time, 1),
            total_climb_m=round(total_climb, 0),
            total_pop_affected=round(total_pop, 0),
        ))

    print(f"  [Router] {len(routes)} route(s) found  ({origin} → {destination})")
    return RouteSet(
        origin=origin,
        destination=destination,
        departure_time=departure_time,
        cruise_kts=cruise_kts,
        wx_filter=wx_filter,
        graph_source=graph_source,
        routes=routes,
        fixes=graph_data.fixes,
    )


# ── Geometry ──────────────────────────────────────────────────────────────────

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Endpoint resolution ────────────────────────────────────────────────────────

def find_nearest_node(graph_data: GraphData, lat: float, lon: float) -> str:
    """Return the ident of the Fix in graph_data closest to (lat, lon)."""
    best_ident = None
    best_dist  = float("inf")
    for fix in graph_data.fixes.values():
        d = haversine_nm(lat, lon, fix.lat, fix.lon)
        if d < best_dist:
            best_dist  = d
            best_ident = fix.ident
    if best_ident is None:
        raise ValueError("graph_data has no fixes")
    return best_ident


def resolve_endpoint(
    name: str,
    graph_data: GraphData,
    airports_csv: str | None = None,
) -> str:
    """
    Resolve an endpoint string to the nearest graph node ident.

    Accepts (tried in order):
    1. "lat,lon"        — coordinate pair, e.g. "36.90,-76.01"
    2. "ORF" / "KORF"   — ICAO airport code → look up in airports.csv
    3. Direct node ident — if name matches a key in graph_data.fixes exactly

    In all cases the result is snapped to the nearest graph node so the caller
    never has to know the internal ident scheme.
    """
    import csv
    from pathlib import Path

    # ── 1. lat,lon pair ───────────────────────────────────────────────────────
    parts = name.strip().split(",")
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            ident = find_nearest_node(graph_data, lat, lon)
            fix   = graph_data.fixes[ident]
            print(f"  [Endpoint] {name!r} → coord snap → {ident!r} "
                  f"({haversine_nm(lat, lon, fix.lat, fix.lon):.1f} nm away)")
            return ident
        except (ValueError, TypeError):
            pass

    # ── 2. ICAO airport lookup ────────────────────────────────────────────────
    icao = name.strip().upper()
    if len(icao) == 3:
        icao = "K" + icao

    csv_candidates = [
        p for p in [
            airports_csv,
            "data/airports.csv",
            "../weatherboy/config/maps/airports.csv",
        ]
        if p and Path(p).exists()
    ]

    for csv_path in csv_candidates:
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("ident", "").upper() == icao or row.get("icao_code", "").upper() == icao:
                        lat = float(row["latitude_deg"])
                        lon = float(row["longitude_deg"])
                        ident = find_nearest_node(graph_data, lat, lon)
                        fix   = graph_data.fixes[ident]
                        print(f"  [Endpoint] {name!r} → airport {icao} "
                              f"({row.get('name','')}) → {ident!r} "
                              f"({haversine_nm(lat, lon, fix.lat, fix.lon):.1f} nm away)")
                        return ident
        except Exception:
            continue

    # ── 3. Direct node ident ──────────────────────────────────────────────────
    if name in graph_data.fixes:
        print(f"  [Endpoint] {name!r} → direct ident match")
        return name

    raise ValueError(
        f"Could not resolve endpoint {name!r}.\n"
        f"Accepted formats:\n"
        f"  lat,lon       e.g. '36.90,-76.01'\n"
        f"  ICAO code     e.g. 'ORF' or 'KORF' (requires airports.csv)\n"
        f"  Node ident    e.g. 'NORFOLK 500KV' (must match graph exactly)"
    )


# ── Live re-weighting ─────────────────────────────────────────────────────────

# Airspace penalty scale (equivalent flight-minutes added to edge cost).
# Hard exclusion = inf.  Class E uses the echo_penalty slider value instead.
AIRSPACE_PENALTY: dict[str, float] = {
    "G":       0.0,
    "E":       500.0,    # default; overridden by echo_penalty slider
    "D":       float("inf"),
    "C":       float("inf"),
    "B":       float("inf"),
    "TFR":     float("inf"),
    "SFRA":    float("inf"),
    "WX":      float("inf"),   # active weather exclusion zone — always hard exclusion
    "UNKNOWN": 0.0,
}

# Severity rank for weather zone threshold comparisons
_WX_SEVERITY_RANK: dict[str, int] = {
    "VFR":  0,
    "MVFR": 1,
    "IFR":  2,
    "LIFR": 3,
}

# wx_rank → hard exclusion threshold (edges at or above this rank are excluded)
WX_HARD_CUTOFF: dict[str, int] = {
    "VFR":  0,
    "MVFR": 1,
    "IFR":  2,
    "ALL":  99,
}


def _ray_cast_simple(lat: float, lon: float, ring) -> bool:
    """Point-in-polygon ray cast for live TFR / weather zone checks.
    ring = [(lat, lon), ...] or [[lat, lon], ...]
    """
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        yi, xi = ring[i][0], ring[i][1]
        yj, xj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def apply_weights(
    G: nx.DiGraph,
    time_weight:                float = 1.0,
    elev_weight:                float = 0.0,
    noise_weight:               float = 0.0,
    cruise_kts:                 float = 120.0,
    wx_filter:                  str   = "MVFR",
    echo_penalty:               float = AIRSPACE_PENALTY["E"],
    airspace_mode:              str   = "default",
    exempt_coords:              list  = None,
    exempt_radius_nm:           float = 25.0,
    current_tfrs:               list  = None,
    current_weather_zones:      list  = None,
    wx_zone_severity_threshold: str   = "IFR",
) -> None:
    """
    Recompute the `weight` attribute on every edge from stored component costs.

    Call this whenever any objective function parameter changes (slider move).
    Modifies G in-place. Fast: O(edges).

    Parameters
    ----------
    time_weight               : λ_t — weight on ETE (default 1.0)
    elev_weight               : λ_e — weight on elevation gain in metres
    noise_weight              : λ_n — population impact weight; activates
                                when pop_sum data is present in graph
    cruise_kts                : current cruise speed (kts); used to compute
                                noise_score = log1p(pop_sum / cruise_kts).
                                Changing this slider immediately shifts the
                                normalised noise surface without a rebuild.
    wx_filter                 : VFR / MVFR / IFR / ALL
    echo_penalty              : equivalent-minute penalty for Class E airspace
    airspace_mode             : reserved
    exempt_coords             : [(lat, lon), ...] — typically origin + destination.
                                Edges within exempt_radius_nm of any of these bypass
                                hard airspace exclusions (not TFR/SFRA/WX).
    exempt_radius_nm          : exemption radius (default 25 nm)
    current_tfrs              : live TFR list — ray-cast against edge midpoints,
                                overrides baked airspace_class to "TFR" (inf weight).
    current_weather_zones     : live WeatherZone list — ray-cast against edge midpoints,
                                sets airspace_class to "WX" (inf weight) for zones at or
                                above wx_zone_severity_threshold. Separate from TFRs so
                                weather and airspace exclusions evolve independently.
    wx_zone_severity_threshold: "MVFR" | "IFR" | "LIFR" — minimum WeatherZone severity
                                that triggers hard exclusion (default "IFR").
    """
    wx_cutoff         = WX_HARD_CUTOFF.get(wx_filter.upper(), 1)
    _threshold_rank   = _WX_SEVERITY_RANK.get(wx_zone_severity_threshold.upper(), 2)

    import datetime as _dt
    _now = _dt.datetime.now(_dt.timezone.utc)

    # Pre-filter active TFRs (avoids per-edge datetime work)
    _active_tfrs: list = []
    if current_tfrs:
        for _tfr in current_tfrs:
            _s = getattr(_tfr, "start_utc", None)
            _e = getattr(_tfr, "end_utc",   None)
            if _s and _now < _s:
                continue
            if _e and _now > _e:
                continue
            _active_tfrs.append(_tfr)

    # Pre-filter active weather zones at or above the severity threshold
    _active_wx_zones: list = []
    if current_weather_zones:
        for _wz in current_weather_zones:
            _s = getattr(_wz, "start_utc", None)
            _e = getattr(_wz, "end_utc",   None)
            if _s and _now < _s:
                continue
            if _e and _now > _e:
                continue
            zone_rank = _WX_SEVERITY_RANK.get(
                getattr(_wz, "severity", "IFR").upper(), 2
            )
            if zone_rank >= _threshold_rank:
                _active_wx_zones.append(_wz)

    # ── Noise score pre-computation (two-pass normalisation) ─────────────────
    # Runs only when noise_weight > 0 AND pop_sum data has been baked into the
    # graph via compute_population_scores() (indicated by pop_sum_max > 0).
    #
    # Pass 1: raw = log1p(pop_sum / cruise_kts) * voltage_multiplier(kv)
    #   - Dividing by cruise_kts converts the spatial population count to a
    #     time-domain exposure proxy: slower flight = more person-seconds.
    #   - voltage_multiplier applies the transmission ROW discount.
    #
    # Pass 2: normalise to [0, 1] across all edges.
    #   - Recomputed each call so cruise_kts changes take effect immediately
    #     without a graph rebuild. Wire is fully exposed for design sweeps.
    if noise_weight > 0.0 and G.graph.get("pop_sum_max", 0.0) > 0.0:
        _raw_noise: dict[tuple, float] = {}
        for _a, _b, _d in G.edges(data=True):
            _ps  = _d.get("pop_sum", 0.0)
            _kv  = _d.get("voltage_kv", 0.0)
            _raw_noise[(_a, _b)] = (
                math.log1p(_ps / max(cruise_kts, 1.0))
                * voltage_multiplier(_kv)
            )
        _nvals = list(_raw_noise.values())
        _nmin  = min(_nvals)
        _nmax  = max(_nvals)
        _nrng  = max(_nmax - _nmin, 1e-9)
        for _a, _b, _d in G.edges(data=True):
            _d["noise_score"] = (_raw_noise[(_a, _b)] - _nmin) / _nrng
    # ─────────────────────────────────────────────────────────────────────────

    for a, b, data in G.edges(data=True):
        ac = data.get("airspace_class", "G")

        # ── Live midpoint checks (TFR + weather zone) ─────────────────────────
        # Compute edge midpoint once; used by both live-override checks.
        if (_active_tfrs or _active_wx_zones) and ac not in ("TFR", "SFRA", "WX"):
            _da   = G.nodes.get(a, {})
            _db   = G.nodes.get(b, {})
            _mlat = (_da.get("lat", 0.0) + _db.get("lat", 0.0)) / 2
            _mlon = (_da.get("lon", 0.0) + _db.get("lon", 0.0)) / 2

            # TFR override — catches TFRs injected after graph build
            if _active_tfrs and ac != "TFR":
                for _tfr in _active_tfrs:
                    if _ray_cast_simple(_mlat, _mlon, _tfr.polygon):
                        ac = "TFR"
                        break

            # Weather zone override — separate from airspace exclusion.
            # WX never exempted by O/D exemption (weather ≠ traffic management).
            if _active_wx_zones and ac not in ("TFR", "SFRA", "WX"):
                for _wz in _active_wx_zones:
                    if _ray_cast_simple(_mlat, _mlon, _wz.polygon):
                        ac = "WX"
                        break

        airspace_pen = echo_penalty if ac == "E" else AIRSPACE_PENALTY.get(ac, 0.0)

        # ── O/D exemption — bypass hard airspace exclusion near endpoints ─────
        # TFR, SFRA, and WX are never exempt.
        if math.isinf(airspace_pen) and ac not in ("TFR", "SFRA", "WX") and exempt_coords:
            da   = G.nodes.get(a, {})
            db   = G.nodes.get(b, {})
            mlat = (da.get("lat", 0.0) + db.get("lat", 0.0)) / 2
            mlon = (da.get("lon", 0.0) + db.get("lon", 0.0)) / 2
            clat = math.radians(mlat)
            for elat, elon in exempt_coords:
                dlat = math.radians(mlat - elat) * 3440.065
                dlon = math.radians(mlon - elon) * 3440.065 * math.cos(clat)
                if math.sqrt(dlat**2 + dlon**2) <= exempt_radius_nm:
                    airspace_pen = 0.0
                    break

        if math.isinf(airspace_pen):
            data["weight"] = float("inf")
            continue

        if data.get("wx_rank", 0) > wx_cutoff:
            data["weight"] = float("inf")
            continue

        data["weight"] = (
            time_weight  * data.get("time_min",    0.0)
            + elev_weight  * data.get("elev_gain_m", 0.0)
            + noise_weight * data.get("noise_score",  0.0)
            + airspace_pen
        )
