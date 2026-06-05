"""
output/analyze_sweep.py — Trailblazer sweep results visualization

Reads a sweep_results.csv (or sensitivity.csv) produced by sweep.py and
generates a two-panel figure:
  - Top:    Population impact (people within 250m buffer) vs noise weight λ_n
  - Bottom: Route distance (nm) vs noise weight λ_n

One line per graph type. Points are averaged across cruise_kts values at each
noise_weight, with shaded ±1σ band where variation exists.

Usage
─────
    cd /path/to/trailblazer
    python output/analyze_sweep.py                          # reads output/sensitivity.csv
    python output/analyze_sweep.py output/sweep_results.csv
    python output/analyze_sweep.py output/sweep_results.csv --save-png output/sweep.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── IBM colorblind-safe palette (mirrors Trailblazer UI) ─────────────────────
IBM = {
    "blue":    "#648FFF",
    "orange":  "#FE6100",
    "green":   "#198038",
    "magenta": "#DC267F",
    "teal":    "#009D9A",
    "gray":    "#A8B2BD",
}

# Graph type → plot color
GRAPH_COLORS = {
    "cell":         IBM["blue"],
    "transmission": IBM["orange"],
}
GRAPH_LABELS = {
    "cell":         "Cell tower mesh",
    "transmission": "Transmission lines",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Trailblazer sweep visualization"
    )
    p.add_argument("csv", nargs="?",
                   default=str(Path(__file__).parent / "sensitivity.csv"),
                   help="Path to sweep results CSV (default: output/sensitivity.csv)")
    p.add_argument("--save-png", default=None, metavar="PATH",
                   help="Save figure to PNG instead of displaying interactively. "
                        "Default: save alongside CSV with same stem.")
    p.add_argument("--k1-only", action="store_true", default=True,
                   help="Plot rank-1 (optimal) routes only (default: True)")
    return p.parse_args(argv)


def _warn(msg: str) -> None:
    print(f"  [analyze] ⚠ {msg}", file=sys.stderr)


def main(argv=None) -> None:
    args = parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # ── Handle inf cost ────────────────────────────────────────────────────
    # inf cost means all routes on that run passed through at least one
    # edge with an infinite airspace penalty (Class D/TFR near O/D).
    # This happens when sweep.py is run without exempt_coords — the route
    # is still found (structural shortest path) but cost is reported as inf.
    # The distance_nm and pop_affected values ARE valid, just cost is not.
    inf_cost_rows = df["cost"].isin([float("inf")]) | df["cost"].apply(
        lambda x: isinstance(x, float) and math.isinf(x)
    )
    # Replace inf with NaN for analysis but keep all rows
    df["cost_clean"] = df["cost"].apply(
        lambda x: float("nan") if (isinstance(x, float) and math.isinf(x)) else x
    )

    inf_pct = inf_cost_rows.sum() / len(df) * 100
    if inf_pct > 0:
        _warn(
            f"{inf_cost_rows.sum()}/{len(df)} rows ({inf_pct:.0f}%) have inf cost. "
            f"This typically means apply_weights() was called without exempt_coords "
            f"in sweep.py — edges near the O/D received infinite airspace penalty. "
            f"Fix: add exempt_coords to sweep.py. "
            f"distance_nm and pop_affected remain valid."
        )

    # ── Filter to rank-1 optimal routes ───────────────────────────────────
    if args.k1_only:
        df = df[df["route_rank"] == 1].copy()

    if df.empty:
        print("[ERROR] No data remaining after filtering.", file=sys.stderr)
        sys.exit(1)

    graphs = sorted(df["graph"].unique())

    # ── Check for constant values (no noise_weight sensitivity) ───────────
    for graph in graphs:
        sub = df[df["graph"] == graph]
        if sub["distance_nm"].std() < 0.01 and sub["pop_affected"].std() < 1.0:
            _warn(
                f"{graph}: distance_nm and pop_affected are constant across all "
                f"noise_weight values — noise_weight is not changing route selection. "
                f"Root cause: likely the exempt_coords issue above. "
                f"Flat lines plotted as-is; they represent a single optimal route."
            )

    # ── Aggregate across cruise_kts at each noise_weight ─────────────────
    # Group by (graph, noise_weight) — mean+std across cruise_kts
    agg = (
        df.groupby(["graph", "noise_weight"])
        .agg(
            dist_mean=("distance_nm",  "mean"),
            dist_std= ("distance_nm",  "std"),
            pop_mean= ("pop_affected",  "mean"),
            pop_std=  ("pop_affected",  "std"),
            cost_mean=("cost_clean",    "mean"),
            n_runs=   ("distance_nm",  "count"),
        )
        .reset_index()
    )
    agg["dist_std"]  = agg["dist_std"].fillna(0)
    agg["pop_std"]   = agg["pop_std"].fillna(0)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    ax_pop, ax_dist = axes
    noise_weights_all = sorted(df["noise_weight"].unique())

    # ── Top panel: population impact ──────────────────────────────────────
    for graph in graphs:
        sub   = agg[agg["graph"] == graph].sort_values("noise_weight")
        color = GRAPH_COLORS.get(graph, IBM["gray"])
        label = GRAPH_LABELS.get(graph, graph)

        ax_pop.plot(
            sub["noise_weight"], sub["pop_mean"],
            color=color, linewidth=2.2, marker="o", markersize=5,
            label=label, zorder=3,
        )
        if sub["pop_std"].max() > 0:
            ax_pop.fill_between(
                sub["noise_weight"],
                sub["pop_mean"] - sub["pop_std"],
                sub["pop_mean"] + sub["pop_std"],
                color=color, alpha=0.12, zorder=2,
            )

    ax_pop.set_ylabel("People within 250 m buffer", fontsize=11)
    ax_pop.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax_pop.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax_pop.grid(axis="y", color="#e0e0e0", linewidth=0.7)
    ax_pop.set_axisbelow(True)
    ax_pop.spines[["top", "right"]].set_visible(False)

    # ── Bottom panel: route distance ───────────────────────────────────────
    for graph in graphs:
        sub   = agg[agg["graph"] == graph].sort_values("noise_weight")
        color = GRAPH_COLORS.get(graph, IBM["gray"])
        label = GRAPH_LABELS.get(graph, graph)

        ax_dist.plot(
            sub["noise_weight"], sub["dist_mean"],
            color=color, linewidth=2.2, marker="o", markersize=5,
            label=label, zorder=3,
        )
        if sub["dist_std"].max() > 0:
            ax_dist.fill_between(
                sub["noise_weight"],
                sub["dist_mean"] - sub["dist_std"],
                sub["dist_mean"] + sub["dist_std"],
                color=color, alpha=0.12, zorder=2,
            )

    ax_dist.set_xlabel("Noise weight  λ_n", fontsize=11)
    ax_dist.set_ylabel("Route distance (nm)", fontsize=11)
    ax_dist.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}"
    ))
    ax_dist.grid(axis="y", color="#e0e0e0", linewidth=0.7)
    ax_dist.set_axisbelow(True)
    ax_dist.spines[["top", "right"]].set_visible(False)

    # x-axis ticks at each noise_weight value
    ax_dist.set_xticks(noise_weights_all)
    ax_dist.set_xticklabels([f"{w:.2f}" for w in noise_weights_all])

    # ── Annotations ───────────────────────────────────────────────────────
    # Inf cost warning banner
    if inf_pct > 0:
        fig.text(
            0.5, 0.985,
            f"⚠  {inf_pct:.0f}% of runs have inf cost (exempt_coords not set in sweep.py) "
            f"— distance and population values remain valid",
            ha="center", va="top", fontsize=8.5,
            color="#7B3F00",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                      edgecolor="#FFC107", linewidth=0.8),
        )

    # Data source note
    meta = df.iloc[0]
    origin_dest = f"{meta.get('origin', '?')} → {meta.get('dest', '?')}"
    n_cruise    = df["cruise_kts"].nunique()
    fig.text(
        0.01, 0.01,
        f"O/D: {origin_dest}  |  "
        f"cruise: {sorted(df['cruise_kts'].unique())} kts  |  "
        f"wx filter: {df['wx_filter'].iloc[0]}  |  "
        f"source: {csv_path.name}",
        fontsize=7.5, color="#777777", va="bottom",
    )

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Trailblazer sensitivity sweep — {origin_dest}",
        fontsize=13, fontweight="bold", y=1.01 if inf_pct > 0 else 0.99,
    )

    # ── Save or show ───────────────────────────────────────────────────────
    if args.save_png:
        out_path = Path(args.save_png)
    else:
        out_path = csv_path.with_suffix(".png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  [analyze] Figure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
