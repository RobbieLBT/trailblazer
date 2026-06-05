"""
cli.py — Trailblazer headless route CLI

Smoke-tests the full pipeline end-to-end without the Streamlit app.
Useful for Phase 1 validation and for scripted batch runs.

Usage
─────
# Transmission graph (Phase 1+)
python -m trailblazer.cli --graph transmission \
    --eia data/eia --origin ORF --dest CRW

# NASR Victor airway graph (comparison)
python -m trailblazer.cli --graph nasr \
    --nasr data/nasr --origin ORF --dest CRW

# Mock weather — demo / portfolio mode
python -m trailblazer.cli --graph transmission \
    --eia data/eia --origin ORF --dest CRW \
    --wx mock --mock-ifr SHD,HCH --mock-mvfr ROA

# Weatherboy integration
python -m trailblazer.cli --graph nasr \
    --nasr data/nasr --origin ORF --dest CRW \
    --wx weatherboy --wb-config ../weatherboy/config/virginia.xml \
    --depart 2026-05-14T14:00:00Z

# Load pre-built graph.pkl (skips parse step — fast)
python -m trailblazer.cli --pkl graph_transmission.pkl \
    --origin ORF --dest CRW

# Export KML for Weatherboy traversal
python cli.py --pkl graph_cell.pkl --origin ORF --dest CRW --write-kml
# → output/ORF_CRW_cell_<date>_rank1.kml
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="trailblazer",
        description="Trailblazer BVLOS route planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Graph source — mutually exclusive: live parse vs pre-built pkl
    src = p.add_mutually_exclusive_group()
    src.add_argument("--graph", default="transmission",
                     choices=["transmission", "nasr", "cell"],
                     help="Build graph live from source data")
    src.add_argument("--pkl", default=None, metavar="PATH",
                     help="Load pre-built graph.pkl (skips parse step)")

    p.add_argument("--eia",  default="data/eia",  metavar="DIR",
                   help="EIA shapefiles dir (--graph transmission)")
    p.add_argument("--fcc",  default="data/fcc",  metavar="DIR",
                   help="FCC ASR dir containing CO.dat (--graph cell)")
    p.add_argument("--nasr", default="data/nasr", metavar="DIR",
                   help="NASR data dir (--graph nasr)")
    p.add_argument("--airports", default=None, metavar="CSV",
                   help="OurAirports airports.csv for ICAO endpoint lookup. "
                        "Auto-detected from data/airports.csv or "
                        "../weatherboy/config/maps/airports.csv")
    p.add_argument("--voltage-min", type=float, default=115.0, metavar="KV",
                   help="Min transmission line voltage (kV)")
    p.add_argument("--bbox", default=None, metavar="LON0,LAT0,LON1,LAT1",
                   help="Bounding box for transmission graph "
                        "(min_lon,min_lat,max_lon,max_lat). "
                        "Default: ORF→CRW AO (-83.5,36.0,-75.5,39.5)")

    p.add_argument("--origin", required=True, metavar="IDENT",
                   help="Origin fix ident")
    p.add_argument("--dest",   required=True, metavar="IDENT",
                   help="Destination fix ident")

    p.add_argument("--wx",      default="gairmet",
                   choices=["gairmet", "weatherboy", "mock", "none"])
    p.add_argument("--wx-filter", default="MVFR",
                   choices=["VFR", "MVFR", "IFR", "LIFR"])
    p.add_argument("--wb-config", default=None, metavar="XML")
    p.add_argument("--mock-ifr",  default=None, metavar="IDENTS")
    p.add_argument("--mock-mvfr", default=None, metavar="IDENTS")

    p.add_argument("--depart", default=None, metavar="ISO8601")
    p.add_argument("--cruise", type=float,   default=120.0,  metavar="KTS")
    p.add_argument("--k",      type=int,     default=3,      metavar="N",
                   help="Number of routes to find")

    p.add_argument("--outdir",     default="output", metavar="DIR")
    p.add_argument("--no-geojson", action="store_true")
    p.add_argument("--no-brief",   action="store_true")
    p.add_argument("--write-kml",  action="store_true", dest="write_kml",
                   help="Also export rank-1 route as KML for Weatherboy "
                        "(written to --outdir alongside GeoJSON and brief).")

    # NASR airway filter (nasr graph only)
    p.add_argument("--airways", default=None, metavar="V268,V20,...",
                   help="Restrict NASR graph to specific airway IDs")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    dep = _parse_departure(args.depart)

    # ── Load graph ─────────────────────────────────────────────────────────────
    if args.pkl:
        G, graph_data = _load_pkl(args.pkl)
        print(f"  [CLI] Loaded graph from {args.pkl} "
              f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    else:
        graph_data = _load_graph_data(args)
        provider   = _build_provider(args, graph_data)

        from trailblazer.routing.pathfinder import build_graph as _build
        print(f"\n[Graph] Building (filter ≥ {args.wx_filter}, "
              f"cruise {args.cruise:.0f} kts, source={graph_data.source})…")
        G = _build(
            graph_data=graph_data,
            wx_provider=provider,
            departure_time=dep,
            cruise_kts=args.cruise,
            wx_filter=args.wx_filter,
        )
        G.graph["wx_filter"]    = args.wx_filter
        G.graph["cruise_kts"]   = args.cruise
        G.graph["graph_source"] = graph_data.source

    # ── Route ──────────────────────────────────────────────────────────────────
    from trailblazer.routing.pathfinder import find_routes, resolve_endpoint

    print(f"\n[Endpoints] Resolving {args.origin!r} and {args.dest!r}…")
    try:
        origin_ident = resolve_endpoint(args.origin, graph_data, args.airports)
        dest_ident   = resolve_endpoint(args.dest,   graph_data, args.airports)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"\n[Router] Finding k={args.k} routes: "
          f"{origin_ident!r} → {dest_ident!r}…")
    try:
        route_set = find_routes(
            G=G,
            origin=origin_ident,
            destination=dest_ident,
            graph_data=graph_data,
            departure_time=dep,
            cruise_kts=args.cruise,
            k=args.k,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # ── Output ─────────────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    stem   = (f"{args.origin.upper()}_{args.dest.upper()}_"
              f"{route_set.graph_source}_{dep.strftime('%Y%m%d_%H%M')}")

    from trailblazer.export.export import write_geojson, write_brief, write_kml
    if not args.no_geojson:
        write_geojson(route_set, outdir / f"{stem}.geojson")
    if not args.no_brief:
        write_brief(route_set, outdir / f"{stem}.md")
    if args.write_kml:
        write_kml(route_set, outdir / f"{stem}_rank1.kml", route_rank=1)

    _print_summary(route_set)
    return 0 if route_set.routes else 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_departure(depart_str: str | None) -> datetime:
    if depart_str:
        try:
            return datetime.fromisoformat(depart_str.replace("Z", "+00:00"))
        except ValueError:
            print(f"[ERROR] Could not parse --depart: {depart_str!r}", file=sys.stderr)
            sys.exit(1)
    dep = datetime.now(tz=timezone.utc)
    print(f"  [CLI] No --depart given, using now: {dep.strftime('%Y-%m-%dT%H%MZ')}")
    return dep


def _load_pkl(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj
    raise ValueError(f"Unexpected pickle format in {path}. Expected (G, GraphData) tuple.")


def _load_graph_data(args):
    if args.graph == "transmission":
        from trailblazer.infra.tx_parser import load_transmission
        print("\n── Loading transmission graph ──")
        bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
        return load_transmission(
            eia_dir=args.eia,
            voltage_min=args.voltage_min,
            bbox=bbox,
        )
    elif args.graph == "cell":
        from trailblazer.infra.tower_parser import load_towers
        bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
        print("\n── Loading cell tower graph ──")
        return load_towers(fcc_dir=args.fcc, bbox=bbox)
    else:  # nasr
        from trailblazer.nasr.parser import load_nasr
        airway_filter = set(args.airways.split(",")) if args.airways else None
        print("\n── Loading NASR graph ──")
        return load_nasr(args.nasr, airway_filter=airway_filter)


def _build_provider(args, graph_data):
    from trailblazer.weather.gairmet import GAirmetProvider
    from trailblazer.weather.provider import MockProvider, CompositeProvider, WeatherboyProvider

    if args.wx == "none":
        print("  [Wx] Provider: MockProvider (all VFR)")
        return MockProvider(default="VFR")

    if args.wx == "mock":
        mock = MockProvider(default="VFR")
        fixes = graph_data.fixes
        for ifr_str, category in ((args.mock_ifr, "IFR"), (args.mock_mvfr, "MVFR")):
            if not ifr_str:
                continue
            for ident in ifr_str.split(","):
                ident = ident.strip().upper()
                fix   = fixes.get(ident)
                if fix:
                    mock.set(ident, category, fix.lat, fix.lon)
                    print(f"  [Wx] Mock {category}: {ident}")
                else:
                    print(f"  [Wx] Warning: mock fix '{ident}' not found in graph data")
        return mock

    if args.wx == "weatherboy":
        if not args.wb_config:
            print("[ERROR] --wb-config required for --wx weatherboy", file=sys.stderr)
            sys.exit(1)
        try:
            import sys as _sys
            import xml.etree.ElementTree as ET
            wb_path = Path(args.wb_config).parent.parent
            if str(wb_path) not in _sys.path:
                _sys.path.insert(0, str(wb_path))
            from weather.fetch import fetch_metars  # noqa — weatherboy module

            tree = ET.parse(args.wb_config)
            root = tree.getroot()
            stations = [el.get("icao") for el in root.findall(".//station") if el.get("icao")]
            if not stations:
                raise ValueError("No <station icao=...> elements found in config")

            from datetime import timedelta
            obs_start = dep - timedelta(hours=1)
            obs_end   = dep + timedelta(hours=6)
            print(f"  [Wx] Fetching Weatherboy METARs for {stations} …")
            obs_data = fetch_metars(stations, obs_start, obs_end)
            print("  [Wx] Provider: Weatherboy + G-AIRMETs (composite)")
            return CompositeProvider([WeatherboyProvider(obs_data), GAirmetProvider()])
        except ImportError as exc:
            print(f"  [Wx] WARNING: Weatherboy import failed ({exc}) — falling back to G-AIRMETs")
            return GAirmetProvider()
        except Exception as exc:
            print(f"  [Wx] WARNING: Weatherboy setup failed ({exc}) — falling back to G-AIRMETs")
            return GAirmetProvider()

    print("  [Wx] Provider: G-AIRMETs (AWC API)")
    return GAirmetProvider()


def _print_summary(route_set) -> None:
    W = 70
    print("\n" + "═" * W)
    print(f"  {route_set.origin} → {route_set.destination}  "
          f"·  {len(route_set.routes)} route(s)  "
          f"·  {route_set.graph_source}  "
          f"·  filter ≥{route_set.wx_filter}  "
          f"·  {route_set.cruise_kts:.0f} kts")
    print("═" * W)

    if not route_set.routes:
        print("  No routes found.")
    else:
        for r in route_set.routes:
            h, m = divmod(int(r.time_min), 60)
            print(f"\n  ── #{r.rank}  {r.distance_nm:.0f} nm  "
                  f"{h}h{m:02d}m ETE  wx: {r.worst_wx}  cost: {r.cost:.1f}")
            for i, fid in enumerate(r.path):
                fix = route_set.fixes.get(fid)
                name      = fix.name  if fix else ""
                connector = "└─" if i == len(r.path) - 1 else "├─"
                print(f"    {connector} {fid:<10} {name:<30}", end="")
                if i < len(r.path) - 1:
                    seg = r.segments[i]
                    kv  = f"{seg.voltage_kv:.0f}kV" if seg.voltage_kv else "off-ROW"
                    print(f"  {seg.airway_id:<8} {kv:<10} "
                          f"[{seg.wx_category}]  {seg.distance_nm:.1f} nm")
                else:
                    print()

    print("\n" + "═" * W)


if __name__ == "__main__":
    sys.exit(main())
