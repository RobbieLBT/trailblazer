"""
sweep.py — Batch parameter sweep for mission space exploration

Loads one or more pre-built graph pkl files and runs find_routes() across
a defined parameter grid, recording route metrics per combination.  Designed
for comparing graph types (cell vs transmission), weight sensitivity, and
cruise speed tradeoffs over a single O/D pair.

Output: results CSV ready for analysis in pandas / any plotting tool.

Usage
─────
# Minimal: compare cell vs transmission at default weights
python sweep.py \
    --pkls graph_cell.pkl graph_transmission.pkl \
    --origin ORF --dest CRW

# Full sensitivity sweep
python sweep.py \
    --pkls graph_cell.pkl graph_transmission.pkl \
    --origin ORF --dest CRW \
    --noise-weights 0.0 0.25 0.5 0.75 1.0 \
    --time-weights  0.5 1.0 1.5 \
    --cruise-kts    80 100 120 140 \
    --k 3 \
    --output output/sweep_results.csv

Monthly weather sensitivity
───────────────────────────
Weather is baked into the pkl at build time, so a true monthly comparison
requires N separate graph builds with different --depart values:

    for DATE in $(seq 1 30); do
        python build_graph.py --graph cell --depart 2026-05-${DATE}T12:00:00Z \
            --landscan data/population/.../landscan-mosaic-unitedstates-v1.tif \
            --outdir output/monthly/
        mv graph_cell.pkl output/monthly/graph_cell_2026-05-${DATE}.pkl
    done

    python sweep.py --pkls output/monthly/graph_cell_*.pkl \
        --origin ORF --dest CRW --output output/monthly_sweep.csv

Then group by date in the output CSV to get monthly distributions.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sweep",
        description="Trailblazer — batch parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pkls", nargs="+", required=True, metavar="PKL",
                   help="One or more graph_*.pkl files to sweep over.")
    p.add_argument("--origin", required=True, metavar="IDENT",
                   help="Origin ICAO / fix ident (e.g. ORF)")
    p.add_argument("--dest", required=True, metavar="IDENT",
                   help="Destination ICAO / fix ident (e.g. CRW)")
    p.add_argument("--airports", default=None, metavar="CSV",
                   help="OurAirports airports.csv for ICAO endpoint lookup.")

    # ── Parameter grid ─────────────────────────────────────────────────────
    p.add_argument("--noise-weights", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0], metavar="W",
                   help="List of noise_weight (λ_n) values to sweep.")
    p.add_argument("--time-weights", nargs="+", type=float,
                   default=[1.0], metavar="W",
                   help="List of time_weight (λ_t) values to sweep.")
    p.add_argument("--elev-weights", nargs="+", type=float,
                   default=[0.0], metavar="W",
                   help="List of elev_weight (λ_e) values to sweep.")
    p.add_argument("--cruise-kts", nargs="+", type=float,
                   default=[120.0], metavar="KTS",
                   help="List of cruise speeds (kts) to sweep.")
    p.add_argument("--wx-filters", nargs="+",
                   default=["MVFR"], metavar="CAT",
                   choices=["VFR", "MVFR", "IFR", "ALL"],
                   help="Weather filter categories to sweep.")
    p.add_argument("--k", type=int, default=1, metavar="N",
                   help="Routes to find per run. k=1 records only the optimal "
                        "route; k>1 records all k routes per parameter set.")

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--output", default="output/sweep_results.csv", metavar="CSV")
    p.add_argument("--no-progress", action="store_true",
                   help="Suppress per-run progress output.")

    return p.parse_args(argv)


def _load_pkl(path: str | Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, tuple) and len(obj) == 2):
        raise ValueError(f"Unexpected pickle format in {path}. Expected (G, GraphData).")
    return obj


def main(argv=None) -> int:
    import pandas as pd
    from trailblazer.routing.pathfinder import (
        apply_weights, find_routes, resolve_endpoint,
    )

    args = parse_args(argv)
    outdir = Path(args.output).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load graphs ────────────────────────────────────────────────────────
    graphs: list[tuple] = []   # (pkl_path, G, graph_data)
    for pkl_path in args.pkls:
        p = Path(pkl_path)
        if not p.exists():
            print(f"[WARN] pkl not found: {p} — skipping", file=sys.stderr)
            continue
        G, gd = _load_pkl(p)
        graphs.append((p, G, gd))
        print(f"  [Sweep] Loaded {p.name}  "
              f"({G.number_of_nodes():,} nodes, {G.number_of_edges()//2:,} edges, "
              f"source={G.graph.get('graph_source', '?')})")

    if not graphs:
        print("[ERROR] No valid pkl files loaded.", file=sys.stderr)
        return 1

    # ── Build parameter grid ───────────────────────────────────────────────
    grid = list(itertools.product(
        args.noise_weights,
        args.time_weights,
        args.elev_weights,
        args.cruise_kts,
        args.wx_filters,
    ))
    total_runs = len(graphs) * len(grid)
    print(f"\n  [Sweep] {len(graphs)} graph(s) × {len(grid)} param combos "
          f"= {total_runs:,} total runs  (k={args.k})\n")

    # ── Run sweep ──────────────────────────────────────────────────────────
    dep   = datetime.now(tz=timezone.utc)
    rows  = []
    run_n = 0
    t0    = time.time()

    for pkl_path, G, graph_data in graphs:
        graph_name   = pkl_path.stem.replace("graph_", "")
        has_pop_data = G.graph.get("pop_sum_max", 0.0) > 0.0

        # Resolve endpoints once per graph
        try:
            origin_id = resolve_endpoint(args.origin, graph_data, args.airports)
            dest_id   = resolve_endpoint(args.dest,   graph_data, args.airports)
        except ValueError as exc:
            print(f"  [Sweep] {graph_name}: endpoint resolution failed — {exc}", file=sys.stderr)
            continue

        # Build exempt_coords from resolved O/D fixes — mirrors the Streamlit
        # app's logic.  Without this, edges in Class D/TFR airspace near the
        # airports get weight=inf, making every route cost inf and preventing
        # noise_weight from changing route selection.
        exempt_coords = []
        for fid in (origin_id, dest_id):
            fix = graph_data.fixes.get(fid)
            if fix and fix.lat is not None and fix.lon is not None:
                exempt_coords.append((fix.lat, fix.lon))

        for noise_w, time_w, elev_w, cruise, wx_filter in grid:
            run_n += 1

            apply_weights(
                G,
                time_weight=time_w,
                elev_weight=elev_w,
                noise_weight=noise_w,
                cruise_kts=cruise,
                wx_filter=wx_filter,
                exempt_coords=exempt_coords or None,
                exempt_radius_nm=10.0,
            )

            route_set = find_routes(
                G, origin_id, dest_id, graph_data,
                departure_time=dep,
                cruise_kts=cruise,
                k=args.k,
            )

            if not args.no_progress:
                n_routes = len(route_set.routes)
                print(f"  [{run_n:>{len(str(total_runs))}}/{total_runs}] "
                      f"{graph_name:<14} "
                      f"λ_n={noise_w:.2f} λ_t={time_w:.2f} λ_e={elev_w:.2f} "
                      f"kts={cruise:.0f} wx={wx_filter:<4}  "
                      f"→ {n_routes} route(s)")

            for route in route_set.routes:
                rows.append({
                    # Identity
                    "graph":          graph_name,
                    "graph_source":   G.graph.get("graph_source", graph_name),
                    "origin":         args.origin,
                    "dest":           args.dest,
                    "origin_node":    origin_id,
                    "dest_node":      dest_id,
                    "has_pop_data":   has_pop_data,
                    # Parameters
                    "noise_weight":   noise_w,
                    "time_weight":    time_w,
                    "elev_weight":    elev_w,
                    "cruise_kts":     cruise,
                    "wx_filter":      wx_filter,
                    "k":              args.k,
                    # Route metrics
                    "route_rank":     route.rank,
                    "distance_nm":    route.distance_nm,
                    "time_min":       route.time_min,
                    "cost":           route.cost,
                    "worst_wx":       route.worst_wx,
                    "waypoints":      len(route.path),
                    "class_e_min":    route.class_e_time_min,
                    "climb_m":        route.total_climb_m,
                    # Social impact
                    "pop_affected":   route.total_pop_affected,
                    "pop_per_nm":     (route.total_pop_affected / route.distance_nm
                                       if route.distance_nm > 0 else 0.0),
                })

    elapsed = time.time() - t0
    print(f"\n  [Sweep] {run_n} runs completed in {elapsed:.1f}s "
          f"({elapsed/max(run_n,1)*1000:.0f} ms/run avg)")

    if not rows:
        print("[WARN] No routes found across any parameter combination.", file=sys.stderr)
        return 2

    # ── Save results ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"  [Sweep] Results → {args.output}  ({len(df):,} rows)")

    # ── Print summary ──────────────────────────────────────────────────────
    _print_summary(df)
    return 0


def _print_summary(df) -> None:
    """Print a compact cross-graph comparison to stdout."""
    import pandas as pd

    W = 78
    print("\n" + "═" * W)
    print("  SWEEP SUMMARY  (k=1 optimal routes only)")
    print("═" * W)

    k1 = df[df["route_rank"] == 1].copy()
    if k1.empty:
        print("  No rank-1 routes to summarise.")
        return

    for graph_name, grp in k1.groupby("graph"):
        print(f"\n  ── {graph_name} ──")
        print(f"     Runs:           {len(grp):,}")
        print(f"     Distance (nm):  {grp['distance_nm'].mean():.1f} avg  "
              f"[{grp['distance_nm'].min():.1f} – {grp['distance_nm'].max():.1f}]")
        print(f"     Time (min):     {grp['time_min'].mean():.1f} avg")
        if grp["pop_affected"].max() > 0:
            print(f"     Pop affected:   {grp['pop_affected'].mean():,.0f} avg  "
                  f"[{grp['pop_affected'].min():,.0f} – {grp['pop_affected'].max():,.0f}]")
            print(f"     Pop/nm:         {grp['pop_per_nm'].mean():,.1f} avg")

    # Cross-graph comparison (if multiple graphs)
    graphs = k1["graph"].unique()
    if len(graphs) == 2:
        g1 = k1[k1["graph"] == graphs[0]]
        g2 = k1[k1["graph"] == graphs[1]]

        # Align on matching parameter combos
        merge_cols = ["noise_weight", "time_weight", "elev_weight",
                      "cruise_kts", "wx_filter"]
        merged = g1[merge_cols + ["distance_nm", "pop_affected"]].merge(
            g2[merge_cols + ["distance_nm", "pop_affected"]],
            on=merge_cols, suffixes=(f"_{graphs[0]}", f"_{graphs[1]}")
        )
        if not merged.empty:
            dist_col_0 = f"distance_nm_{graphs[0]}"
            dist_col_1 = f"distance_nm_{graphs[1]}"
            pop_col_0  = f"pop_affected_{graphs[0]}"
            pop_col_1  = f"pop_affected_{graphs[1]}"

            print(f"\n  ── {graphs[0]} vs {graphs[1]} (matched param pairs: {len(merged)}) ──")
            d_diff = (merged[dist_col_0] - merged[dist_col_1]).mean()
            print(f"     Distance delta:  {d_diff:+.1f} nm avg "
                  f"({'shorter' if d_diff < 0 else 'longer'} for {graphs[0]})")
            if merged[pop_col_0].max() > 0 or merged[pop_col_1].max() > 0:
                p_diff = (merged[pop_col_0] - merged[pop_col_1]).mean()
                print(f"     Pop delta:       {p_diff:+,.0f} avg "
                      f"({'fewer' if p_diff < 0 else 'more'} people for {graphs[0]})")

    print("\n" + "═" * W)


if __name__ == "__main__":
    sys.exit(main())
