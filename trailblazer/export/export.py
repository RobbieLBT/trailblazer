"""
trailblazer/export/export.py — GeoJSON, KML, and mission brief outputs

Four output formats from a single RouteSet:

  write_geojson()   GeoJSON FeatureCollection for QGIS / geojson.io / Leaflet
  write_kml()       KML LineString for Weatherboy mission traversal
  write_brief()     Markdown mission brief for human review / PDF conversion
  to_dict()         Structured dict for downstream module consumption
                    (energy model, noise model, SIL harness, Streamlit)

Weatherboy KML handoff
──────────────────────
write_kml() produces a KML LineString compatible with Weatherboy's
mission/path.py parser.  Pass the output path to weatherboy via:

    python3 run.py --config config/virginia.xml \\
                   --path output/ORF_CRW_cell_rank1.kml \\
                   --alt-agl 350 --speed-kmh 65 \\
                   --traverse-dt 10 \\
                   --output output/ORF_CRW_cell_rank1_forcing.csv \\
                   --no-animate

The resulting forcing CSV contains ForcingRecord fields (wind_ned, gust_ned,
headwind_ms, crosswind_ms, delta_beta_deg, density_kgm3, flight_category) at
--traverse-dt second intervals along the route — ready for Phase 3 vehicle
dynamics integration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..routing.pathfinder import RouteSet, Route


_ROUTE_COLORS = ["#1E90FF", "#FF8C00", "#3CB371", "#9B59B6", "#E74C3C"]
_FC_SYMBOL    = {"VFR": "✅", "MVFR": "🔵", "IFR": "🔴", "LIFR": "🟣", "UNKNOWN": "⚪"}


# ── GeoJSON ───────────────────────────────────────────────────────────────────

def write_geojson(route_set: RouteSet, path: str | Path) -> Path:
    """
    Write a GeoJSON FeatureCollection.

    Features:
      One LineString per route, styled for geojson.io / Mapbox GL
      One Point per unique fix on any route
    """
    path = Path(path)
    features = []
    dep_iso = route_set.departure_time.isoformat()

    for r in route_set.routes:
        color = _ROUTE_COLORS[min(r.rank - 1, len(_ROUTE_COLORS) - 1)]
        coords = [
            [route_set.fixes[fid].lon, route_set.fixes[fid].lat]
            for fid in r.path
            if fid in route_set.fixes
        ]
        h, m = divmod(int(r.time_min), 60)

        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "route_rank":    r.rank,
                "origin":        route_set.origin,
                "destination":   route_set.destination,
                "departure_utc": dep_iso,
                "graph_source":  route_set.graph_source,
                "distance_nm":   r.distance_nm,
                "time_min":      r.time_min,
                "ete":           f"{h}h{m:02d}m",
                "worst_wx":      r.worst_wx,
                "airways":       ", ".join(r.unique_airways),
                "path":          " → ".join(r.path),
                "cost":          r.cost,
                "stroke":        color,
                "stroke-width":  max(1, 4 - r.rank),
                "stroke-opacity": 0.85,
            },
        })

    seen_fixes: set[str] = set()
    for r in route_set.routes:
        for fid in r.path:
            if fid in seen_fixes or fid not in route_set.fixes:
                continue
            seen_fixes.add(fid)
            fix = route_set.fixes[fid]
            is_endpoint = fid in (route_set.origin, route_set.destination)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [fix.lon, fix.lat]},
                "properties": {
                    "ident":         fid,
                    "name":          fix.name,
                    "fix_type":      fix.fix_type,
                    "state":         fix.state,
                    "marker-symbol": "airport" if is_endpoint else "triangle",
                    "marker-size":   "large"   if is_endpoint else "small",
                    "marker-color":  "#E53935" if is_endpoint else "#555555",
                },
            })

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
            "origin":        route_set.origin,
            "destination":   route_set.destination,
            "graph_source":  route_set.graph_source,
            "wx_filter":     route_set.wx_filter,
            "cruise_kts":    route_set.cruise_kts,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"  [Export] GeoJSON → {path}")
    return path


# ── KML (Weatherboy mission traversal format) ─────────────────────────────────

def kml_string(
    route_set: RouteSet,
    route_rank: int | None = 1,
) -> str:
    """
    Build and return the KML document as a string without writing a file.

    Use this when you need the content in memory — e.g. a Streamlit
    download button.  write_kml() calls this internally.
    """
    return _build_kml_string(route_set, route_rank)


def write_kml(
    route_set: RouteSet,
    path: str | Path,
    route_rank: int | None = 1,
) -> Path:
    """
    Write one or more routes as KML LineStrings for Weatherboy mission traversal.

    Produces a KML 2.2 Document whose LineString format is identical to
    Weatherboy's VA-XC1.kml reference — parsed by mission/path.py.

    Parameters
    ----------
    route_set  : RouteSet from find_routes()
    path       : output file path (e.g. "output/ORF_CRW_cell_rank1.kml")
    route_rank : which route to export.  1 = optimal route (default).
                 None = export all routes as separate Placemarks in one file.

    Notes
    ─────
    Coordinates are written as "lon,lat,0" tuples (KML convention: lon first).
    Altitude is MSL=0 — Weatherboy applies --alt-agl at traversal time.
    """
    kml = _build_kml_string(route_set, route_rank)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kml, encoding="utf-8")
    n = route_rank if route_rank else len(route_set.routes)
    print(f"  [Export] KML → {path}  ({n} route(s))")
    return path


def _build_kml_string(
    route_set: RouteSet,
    route_rank: int | None = 1,
) -> str:
    """Internal: build the KML document string (shared by kml_string + write_kml)."""

    # Select routes to export
    if route_rank is not None:
        routes = [r for r in route_set.routes if r.rank == route_rank]
        if not routes:
            raise ValueError(
                f"Route rank {route_rank} not found in RouteSet "
                f"(available: {[r.rank for r in route_set.routes]})"
            )
    else:
        routes = route_set.routes

    dep_iso = route_set.departure_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── KML color encoding per route rank ─────────────────────────────────
    # KML color format: aabbggrr (alpha, blue, green, red — reversed from CSS)
    _KML_COLORS = [
        "ffFF8C1E",   # rank 1: blue  (#1E90FF reversed)
        "ff00008CFF", # rank 2: orange
        "ff71B33C",   # rank 3: green
    ]

    # ── Build Placemark blocks ─────────────────────────────────────────────
    placemarks: list[str] = []

    for r in routes:
        coords = [
            route_set.fixes[fid]
            for fid in r.path
            if fid in route_set.fixes
        ]
        if len(coords) < 2:
            continue

        coord_str = " ".join(
            f"{fix.lon:.8f},{fix.lat:.8f},0"
            for fix in coords
        )

        kml_color = _KML_COLORS[min(r.rank - 1, len(_KML_COLORS) - 1)]
        h, m      = divmod(int(r.time_min), 60)
        style_id  = f"route{r.rank}Style"

        placemarks.append(f"""
\t<Style id="{style_id}_normal">
\t\t<LineStyle>
\t\t\t<color>{kml_color}</color>
\t\t\t<width>{max(2, 5 - r.rank)}</width>
\t\t</LineStyle>
\t\t<PolyStyle><color>40ffffff</color></PolyStyle>
\t</Style>
\t<Style id="{style_id}_highlight">
\t\t<LineStyle>
\t\t\t<color>{kml_color}</color>
\t\t\t<width>{max(3, 6 - r.rank)}</width>
\t\t</LineStyle>
\t</Style>
\t<StyleMap id="{style_id}">
\t\t<Pair><key>normal</key><styleUrl>#{style_id}_normal</styleUrl></Pair>
\t\t<Pair><key>highlight</key><styleUrl>#{style_id}_highlight</styleUrl></Pair>
\t</StyleMap>
\t<Placemark>
\t\t<name>{route_set.origin}→{route_set.destination} #{r.rank} ({r.distance_nm:.0f}nm {h}h{m:02d}m)</name>
\t\t<description>Graph: {route_set.graph_source} | Depart: {dep_iso} | Cruise: {route_set.cruise_kts:.0f}kts | Worst wx: {r.worst_wx} | Cost: {r.cost:.1f}</description>
\t\t<styleUrl>#{style_id}</styleUrl>
\t\t<LineString>
\t\t\t<tessellate>1</tessellate>
\t\t\t<coordinates>
\t\t\t\t{coord_str}
\t\t\t</coordinates>
\t\t</LineString>
\t</Placemark>""")

    doc_name = (
        f"{route_set.origin}-{route_set.destination}-{route_set.graph_source}"
        f"-rank{route_rank}" if route_rank else
        f"{route_set.origin}-{route_set.destination}-{route_set.graph_source}-all"
    )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2"
     xmlns:kml="http://www.opengis.net/kml/2.2">
<Document id="document">
\t<name>{doc_name}</name>
\t<description>Generated by Trailblazer {datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")} | {route_set.origin} → {route_set.destination} | {route_set.graph_source} | ≥{route_set.wx_filter}</description>
{"".join(placemarks)}
</Document>
</kml>
"""

    return kml


# ── Mission brief (Markdown) ──────────────────────────────────────────────────

def write_brief(route_set: RouteSet, path: str | Path) -> Path:
    """Write a human-readable Markdown mission brief."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dep = route_set.departure_time
    dep_str = (dep.strftime("%Y-%m-%d %H%MZ") if dep.tzinfo
               else dep.strftime("%Y-%m-%d %H%M local"))

    lines: list[str] = [
        f"# Route Analysis — {route_set.origin} → {route_set.destination}",
        "",
        f"**Departure:** {dep_str}  ",
        f"**Cruise speed:** {route_set.cruise_kts:.0f} kts  ",
        f"**Graph source:** {route_set.graph_source}  ",
        f"**Weather filter:** ≥ {route_set.wx_filter}  ",
        f"**Generated:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H%MZ')}  ",
        "",
    ]

    if not route_set.routes:
        lines += [
            "## No viable routes found",
            "",
            f"No routes from {route_set.origin} to {route_set.destination} satisfy "
            f"the ≥{route_set.wx_filter} weather filter with the current picture.",
            "",
            "**Suggested actions:**",
            "- Check individual fix weather conditions",
            "- Relax weather filter to IFR (if IFR-capable)",
            "- Delay departure and re-run with updated forecast",
            "",
        ]
    else:
        lines += [
            "## Route Summary",
            "",
            "| Rank | Distance | ETE | Airways | Worst Wx | Cost |",
            "|------|----------|-----|---------|----------|------|",
        ]
        for r in route_set.routes:
            h, m   = divmod(int(r.time_min), 60)
            symbol = _FC_SYMBOL.get(r.worst_wx, "⚪")
            lines.append(
                f"| #{r.rank} | {r.distance_nm:.0f} nm | {h}h{m:02d}m "
                f"| {', '.join(r.unique_airways)} | {symbol} {r.worst_wx} | {r.cost:.1f} |"
            )
        lines.append("")

        for r in route_set.routes:
            h, m = divmod(int(r.time_min), 60)
            lines += [
                "---",
                "",
                f"## Route #{r.rank}",
                "",
                f"**{r.distance_nm:.0f} nm · {h}h{m:02d}m ETE · "
                f"{_FC_SYMBOL.get(r.worst_wx,'')} {r.worst_wx} worst-case**",
                "",
                "| Leg | From | To | Corridor | kV | Dist (nm) | ETE (min) | Wx |",
                "|-----|------|----|----------|----|-----------|-----------|----|",
            ]
            for i, seg in enumerate(r.segments):
                fa = route_set.fixes.get(seg.from_ident)
                fb = route_set.fixes.get(seg.to_ident)
                from_name = f"{seg.from_ident} {fa.name}" if fa else seg.from_ident
                to_name   = f"{seg.to_ident} {fb.name}"   if fb else seg.to_ident
                symbol = _FC_SYMBOL.get(seg.wx_category, "⚪")
                kv_str = f"{seg.voltage_kv:.0f}" if seg.voltage_kv else "—"
                lines.append(
                    f"| {i+1} | {from_name} | {to_name} | {seg.airway_id} "
                    f"| {kv_str} | {seg.distance_nm:.1f} | {seg.time_min:.1f} "
                    f"| {symbol} {seg.wx_category} |"
                )
            lines.append("")

    path.write_text("\n".join(lines))
    print(f"  [Export] Mission brief → {path}")
    return path


# ── Structured dict ───────────────────────────────────────────────────────────

def to_dict(route_set: RouteSet) -> dict[str, Any]:
    """
    Serialise a RouteSet to a plain Python dict for downstream modules.

    Schema mirrors the handoff spec with voltage_kv added per segment.
    """
    dep_iso     = route_set.departure_time.isoformat()
    routes_out  = []

    for r in route_set.routes:
        segs_out = []
        for s in r.segments:
            fa = route_set.fixes.get(s.from_ident)
            fb = route_set.fixes.get(s.to_ident)
            segs_out.append({
                "from":        s.from_ident,
                "to":          s.to_ident,
                "airway":      s.airway_id,
                "distance_nm": s.distance_nm,
                "time_min":    s.time_min,
                "wx":          s.wx_category,
                "voltage_kv":  s.voltage_kv,
                "from_lat":    fa.lat if fa else None,
                "from_lon":    fa.lon if fa else None,
                "to_lat":      fb.lat if fb else None,
                "to_lon":      fb.lon if fb else None,
            })
        routes_out.append({
            "rank":             r.rank,
            "distance_nm":      r.distance_nm,
            "time_min":         r.time_min,
            "worst_wx":         r.worst_wx,
            "cost":             r.cost,
            "path":             r.path,
            "segments":         segs_out,
            "total_pop_affected": r.total_pop_affected,
        })

    return {
        "origin":         route_set.origin,
        "destination":    route_set.destination,
        "departure_utc":  dep_iso,
        "cruise_kts":     route_set.cruise_kts,
        "graph_source":   route_set.graph_source,
        "wx_filter":      route_set.wx_filter,
        "routes":         routes_out,
    }
