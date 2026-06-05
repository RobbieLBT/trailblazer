"""
build_graph.py — Stage 1 entry point

Pulls and filters all data sources, builds the weighted NetworkX graph,
and serialises it to graph.pkl for the Streamlit app (Stage 2).

This is the expensive step — EIA shapefiles, population raster, FCC ASR.
Run it once; the Streamlit app loads the cached graph in under a second.

Usage
─────
# Transmission line graph at 350 ft AGL
python build_graph.py --graph transmission --airspace --operating-alt 350 --elevation

# With elevation cache (default on — avoids re-fetching on rebuild)
python build_graph.py --graph transmission --elevation --elev-cache data/elevation_cache.json

# Skip elevation cache (force re-fetch)
python build_graph.py --graph transmission --elevation --no-elev-cache

# Denser cell mesh
python build_graph.py --graph cell --cell-grid-km 12.5 --cell-max-edge 40 --elevation

# Cell mesh with population scoring
python build_graph.py --graph cell --elevation --airspace --operating-alt 350 \
    --landscan data/population/landscan-mosaic-unitedstates-v1-assets/landscan-mosaic-unitedstates-v1.tif
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_graph",
        description="Trailblazer — build and cache weighted routing graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", default="transmission",
                   choices=["transmission", "nasr", "cell", "both"],
                   help="Graph source to build. 'both' builds transmission+nasr.")
    p.add_argument("--eia",   default="data/eia",   metavar="DIR")
    p.add_argument("--fcc",   default="data/fcc",   metavar="DIR")
    p.add_argument("--nasr",  default="data/nasr",  metavar="DIR")
    p.add_argument("--outdir", default=".",          metavar="DIR")

    p.add_argument("--wx", default="gairmet",
                   choices=["gairmet", "weatherboy", "mock", "none"])
    p.add_argument("--wx-filter", default="MVFR",
                   choices=["VFR", "MVFR", "IFR", "LIFR"])
    p.add_argument("--wb-config", default=None, metavar="XML")
    p.add_argument("--mock-ifr",  default=None, metavar="IDENTS")

    p.add_argument("--depart", default=None, metavar="ISO8601")
    p.add_argument("--cruise", type=float, default=120.0, metavar="KTS")
    p.add_argument("--voltage-min", type=float, default=115.0, metavar="KV")

    p.add_argument("--airspace", action="store_true",
                   help="Classify edges by airspace (requires NASR class_airspace shapefiles)")
    p.add_argument("--operating-alt", type=float, default=400.0, metavar="FT_AGL",
                   dest="operating_alt",
                   help="Operating altitude (ft AGL) for Class E altitude filter. "
                        "Default 400. Use 350 for this mission.")
    p.add_argument("--tfrs", default=None, metavar="JSON")

    p.add_argument("--elevation", action="store_true",
                   help="Fetch/load terrain elevation for all nodes via Open-Topo-Data.")
    p.add_argument("--elev-cache", default="data/elevation_cache.json",
                   metavar="JSON", dest="elev_cache",
                   help="Path to the elevation disk cache.  Populated on first fetch; "
                        "subsequent builds load from it instantly.  "
                        "Default: data/elevation_cache.json")
    p.add_argument("--no-elev-cache", action="store_true", dest="no_elev_cache",
                   help="Disable the elevation cache — always fetch from API.")
    p.add_argument("--elev-weight", type=float, default=0.0, metavar="W")

    p.add_argument("--tiles", action="store_true", default=True)
    p.add_argument("--no-tiles", dest="tiles", action="store_false")
    p.add_argument("--tile-zoom-max", type=int, default=11, metavar="Z")

    p.add_argument("--bbox", default=None, metavar="LON0,LAT0,LON1,LAT1")

    # ── Cell tower mesh ────────────────────────────────────────────────────
    p.add_argument("--cell-grid-km", type=float, default=25.0, metavar="KM",
                   dest="cell_grid_km",
                   help="Grid reduction cell size for cell tower mesh (km). "
                        "Halving roughly 4x the node count. Default 25.0.")
    p.add_argument("--cell-max-edge", type=float, default=60.0, metavar="KM",
                   dest="cell_max_edge",
                   help="Max Delaunay edge length for cell tower mesh (km). "
                        "Default 60.0. Reduce to 40 when using a denser grid.")

    # ── Population scoring ─────────────────────────────────────────────────
    p.add_argument("--landscan", default=None, metavar="TIF",
                   help="Path to LandScan USA GeoTIFF for population impact "
                        "scoring. Stores raw pop_sum on every edge at build time; "
                        "the Noise λ_n slider in the app activates it. "
                        "Example: data/population/landscan-mosaic-unitedstates"
                        "-v1-assets/landscan-mosaic-unitedstates-v1.tif")

    return p.parse_args(argv)


def _load_graph_data(args):
    graph_data_map = {}

    if args.graph in ("transmission", "both"):
        from trailblazer.infra.tx_parser import load_transmission
        print("\n── Loading transmission graph ──")
        bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
        gd = load_transmission(eia_dir=args.eia, voltage_min=args.voltage_min, bbox=bbox)
        graph_data_map["transmission"] = gd

    if args.graph in ("nasr", "both"):
        from trailblazer.nasr.parser import load_nasr
        print("\n── Loading NASR graph ──")
        graph_data_map["nasr"] = load_nasr(args.nasr)

    if args.graph == "cell":
        from trailblazer.infra.tower_parser import load_towers
        print("\n── Loading cell tower graph ──")
        bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
        gd = load_towers(
            fcc_dir=args.fcc,
            bbox=bbox,
            grid_cell_km=args.cell_grid_km,
            max_edge_km=args.cell_max_edge,
        )
        graph_data_map["cell"] = gd

    return graph_data_map


def _build_provider(args, graph_data=None, dep=None):
    from trailblazer.weather.gairmet import GAirmetProvider
    from trailblazer.weather.provider import MockProvider, CompositeProvider, WeatherboyProvider

    if args.wx == "none":
        print("  [Wx] Provider: MockProvider (all VFR)")
        return MockProvider(default="VFR")

    if args.wx == "mock":
        mock = MockProvider(default="VFR")
        if args.mock_ifr and graph_data:
            fixes = list(graph_data.values())[0].fixes
            for ident in args.mock_ifr.split(","):
                ident = ident.strip().upper()
                fix = fixes.get(ident)
                if fix:
                    mock.set(ident, "IFR", fix.lat, fix.lon)
                    print(f"  [Wx] Mock IFR: {ident}")
        return mock

    if args.wx == "weatherboy":
        if not args.wb_config:
            print("[ERROR] --wb-config required for --wx weatherboy", file=sys.stderr)
            sys.exit(1)
        try:
            import sys as _sys
            import xml.etree.ElementTree as ET
            from datetime import timedelta
            wb_path = Path(args.wb_config).parent.parent
            if str(wb_path) not in _sys.path:
                _sys.path.insert(0, str(wb_path))
            from weather.fetch import fetch_metars  # noqa

            tree = ET.parse(args.wb_config)
            root = tree.getroot()
            stations = [el.get("icao") for el in root.findall(".//station") if el.get("icao")]
            if not stations:
                raise ValueError("No <station icao=...> elements found in config")

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


def main(argv=None) -> int:
    args = parse_args(argv)

    dep = _parse_departure(args.depart)

    # Resolve elevation cache path (None = disabled)
    elev_cache: Path | None = None
    if args.elevation and not args.no_elev_cache:
        elev_cache = Path(args.elev_cache)

    print(f"\n[build_graph] departure={dep.strftime('%Y-%m-%dT%H%MZ')} "
          f"cruise={args.cruise:.0f}kts wx={args.wx} filter≥{args.wx_filter} "
          f"operating_alt={args.operating_alt:.0f}ft")
    if args.graph == "cell":
        print(f"  cell: grid_km={args.cell_grid_km}  max_edge_km={args.cell_max_edge}")
    if args.elevation:
        print(f"  elev_cache: {elev_cache or 'disabled'}")
    if args.landscan:
        print(f"  landscan: {args.landscan}")

    graph_data_map = _load_graph_data(args)
    if not graph_data_map:
        print("[ERROR] No graph data loaded.", file=sys.stderr)
        return 1

    provider = _build_provider(args, graph_data_map, dep=dep)

    from trailblazer.routing.pathfinder import build_graph

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, graph_data in graph_data_map.items():
        airspace_filter = None
        if args.airspace:
            from trailblazer.airspace.exclusion import load_airspace
            bbox_t = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
            airspace_filter = load_airspace(
                tfrs_json=args.tfrs,
                bbox=bbox_t,
                operating_agl_ft=args.operating_alt,
            )

        if args.elevation:
            from trailblazer.elevation import fetch_elevations
            print(f"\n── Fetching terrain elevations for {name} graph ──")
            fetch_elevations(graph_data, cache_path=elev_cache)

        print(f"\n── Building {name} graph ──")
        G = build_graph(
            graph_data=graph_data,
            wx_provider=provider,
            departure_time=dep,
            cruise_kts=args.cruise,
            wx_filter=args.wx_filter,
            elev_weight=args.elev_weight,
            airspace_filter=airspace_filter,
        )

        if args.landscan:
            from trailblazer.scoring.social import compute_population_scores, generate_population_overlay
            compute_population_scores(G, args.landscan, buffer_m=250.0)
            overlay_png  = outdir / f"population_{name}.png"
            overlay_meta = outdir / f"population_{name}.json"
            generate_population_overlay(G, args.landscan, overlay_png, overlay_meta)

        G.graph["wx_filter"]    = args.wx_filter
        G.graph["cruise_kts"]   = args.cruise
        G.graph["graph_source"] = name

        out_path = outdir / f"graph_{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump((G, graph_data), f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  [Build] Saved → {out_path} "
              f"({out_path.stat().st_size / 1_048_576:.1f} MB)")

        if args.tiles:
            print(f"\n── Generating tiles for {name} ──")
            from trailblazer.tiles import generate_tiles
            tiles_out = Path(__file__).parent / "app" / "static" / "tiles"
            generate_tiles(G, out_dir=tiles_out, name=name, zoom_max=args.tile_zoom_max)
        else:
            print(f"  [Tiles] Skipped (--no-tiles)")

    print("\n[build_graph] Done.  Run the Streamlit app with:")
    print("  streamlit run app/streamlit_app.py")
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
