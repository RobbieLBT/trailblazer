"""
trailblazer/tiles.py — Render graph edges as a PNG XYZ tile pyramid.

Tiles are written to:
    app/static/tiles/{graph_name}/{z}/{x}/{y}.png

Streamlit serves this directory at:
    http://localhost:<port>/app/static/tiles/...
(requires  server.enableStaticServing = true  in .streamlit/config.toml)

The tile layer replaces the Folium GeoJSON edge layer, moving all edge
rendering off the WebSocket and into the browser's normal tile-fetching path.
Browser fetches a few hundred KB of PNG per zoom level instead of 500 MB of
embedded JavaScript.

Visual encoding
───────────────
All edges use a single colour (IBM cyan #33B1FF) so voltage classes don't
compete visually with airspace overlays and route lines.  Voltage is encoded
by line weight:  765 kV is the thickest, 115 kV the thinnest.  Cell tower
mesh edges use a slightly thinner weight than 115 kV transmission lines so
the two graph types remain distinguishable.

Usage
─────
Called automatically by build_graph.py --tiles (default on).
Can also be called directly:

    from trailblazer.tiles import generate_tiles
    generate_tiles(G, out_dir=Path("app/static/tiles"), name="transmission")

Dependencies: mercantile, matplotlib, numpy  (all in requirements.txt)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                          # no display needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mercantile
import numpy as np

# ── Tile parameters ───────────────────────────────────────────────────────────

ZOOM_MIN  = 6    # continental overview
ZOOM_MAX  = 11   # ~75 m/px at Virginia latitudes — individual line detail
TILE_PX   = 256
DPI       = 96   # → figsize = 256/96 ≈ 2.667 in (exactly 256 px output)

# ── Unified colour — IBM cyan (matches TOWER_MESH in streamlit_app.py) ────────
# All edges use the same hue; voltage is encoded by line weight only.
# This keeps the network layer visually subordinate to route lines and
# airspace overlays (which use the full IBM colorblind-safe palette).

_EDGE_COLOR  = "#33B1FF"   # IBM cyan
_DEFAULT_COLOR = _EDGE_COLOR

# Base line widths in points at z8; scaled at higher zooms.
# Tower mesh is slightly thinner than 115 kV so the two layers read separately.
_BASE_LW: dict[str, float] = {
    "765KV":      3.5,
    "500KV":      2.8,
    "345KV":      2.2,
    "230KV":      1.6,
    "115KV":      1.1,
    "TXLOW":      0.7,
    "TOWER_MESH": 0.9,
}
_DEFAULT_LW = 0.8
_ALPHA      = 0.65


def _edge_color(_airway: str) -> str:
    """All edges use the unified IBM cyan colour."""
    return _EDGE_COLOR


def _line_width(airway: str, zoom: int) -> float:
    base = _BASE_LW.get(airway.upper(), _DEFAULT_LW)
    # Thicken slightly at higher zooms (lines are zoomed in, need to stay crisp)
    return base * (1.0 + max(0, zoom - 8) * 0.2)


def _to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """WGS-84 lon/lat → Web Mercator (EPSG:3857) in metres."""
    R = 6378137.0
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * R
    return x, y


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_tiles(
    G,
    out_dir: Path,
    name: str    = "graph",
    zoom_min: int = ZOOM_MIN,
    zoom_max: int = ZOOM_MAX,
) -> Path:
    """
    Render all graph edges as a PNG tile pyramid and write to disk.

    Parameters
    ----------
    G        : NetworkX graph — nodes must have 'lat' and 'lon' attributes.
    out_dir  : Root output directory.  Tiles land at out_dir/name/{z}/{x}/{y}.png.
    name     : Graph name used as tile subdirectory (e.g. "transmission", "cell").
    zoom_min : Lowest zoom level to generate (default 6).
    zoom_max : Highest zoom level to generate (default 11).

    Returns
    -------
    Path to tile root (out_dir/name/).
    """
    tile_root = Path(out_dir) / name
    tile_root.mkdir(parents=True, exist_ok=True)

    # ── Project all edges to Web Mercator ─────────────────────────────────────
    print(f"  [Tiles] Projecting {name} edges to Web Mercator…")
    records: list[tuple] = []
    seen: set[frozenset] = set()

    for a, b, data in G.edges(data=True):
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        na = G.nodes.get(a, {})
        nb = G.nodes.get(b, {})
        if "lat" not in na or "lat" not in nb:
            continue
        airway = data.get("airway", "")
        mx1, my1 = _to_mercator(na["lon"], na["lat"])
        mx2, my2 = _to_mercator(nb["lon"], nb["lat"])
        # Store (x1, y1, x2, y2, color, airway) — color is uniform but kept
        # in the tuple so the group-by-key loop below stays unchanged.
        records.append((mx1, my1, mx2, my2, _edge_color(airway), airway))

    if not records:
        print("  [Tiles] No edges found — nothing to render.")
        return tile_root

    print(f"  [Tiles] {len(records):,} edges ready")

    # Vectorised bounding-box arrays for fast tile-intersection tests
    arr  = np.array([(r[0], r[1], r[2], r[3]) for r in records], dtype=np.float64)
    xmin = np.minimum(arr[:, 0], arr[:, 2])
    xmax = np.maximum(arr[:, 0], arr[:, 2])
    ymin = np.minimum(arr[:, 1], arr[:, 3])
    ymax = np.maximum(arr[:, 1], arr[:, 3])

    # Bounding box in lon/lat for mercantile.tiles()
    node_lons = [G.nodes[n]["lon"] for n in G.nodes if "lon" in G.nodes[n]]
    node_lats = [G.nodes[n]["lat"] for n in G.nodes if "lat" in G.nodes[n]]
    if not node_lons:
        print("  [Tiles] No node coordinates — aborting.")
        return tile_root
    bbox_ll = (min(node_lons), min(node_lats), max(node_lons), max(node_lats))

    figsize  = TILE_PX / DPI          # square tile in inches
    total_written = 0
    total_empty   = 0

    for zoom in range(zoom_min, zoom_max + 1):
        zoom_tiles = list(mercantile.tiles(*bbox_ll, zooms=zoom))
        written    = 0
        print(f"  [Tiles] Zoom {zoom:2d}  ({len(zoom_tiles):5,} candidates)…",
              end="", flush=True)

        for tile in zoom_tiles:
            bnd = mercantile.xy_bounds(tile)   # Web Mercator bounds of this tile

            # Fast numpy intersection: skip edges whose bbox doesn't touch the tile
            mask = ~(
                (xmax < bnd.left)  |
                (xmin > bnd.right) |
                (ymax < bnd.bottom)|
                (ymin > bnd.top)
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                total_empty += 1
                continue

            # ── Render tile ───────────────────────────────────────────────
            fig = plt.figure(figsize=(figsize, figsize), dpi=DPI)
            ax  = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(bnd.left,   bnd.right)
            ax.set_ylim(bnd.bottom, bnd.top)
            ax.axis("off")
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

            # Group by linewidth only (colour is uniform, so groups collapse
            # by weight class — fewer LineCollection objects per tile).
            groups: dict[float, list] = {}
            for i in idx:
                r  = records[i]
                lw = round(_line_width(r[5], zoom), 2)
                if lw not in groups:
                    groups[lw] = []
                groups[lw].append([(r[0], r[1]), (r[2], r[3])])

            for lw, segs in groups.items():
                ax.add_collection(
                    LineCollection(segs, colors=_EDGE_COLOR, linewidths=lw,
                                   alpha=_ALPHA, capstyle="round")
                )

            # Write transparent PNG
            out_path = tile_root / str(zoom) / str(tile.x)
            out_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out_path / f"{tile.y}.png",
                transparent=True, dpi=DPI,
                bbox_inches=None, pad_inches=0,
            )
            plt.close(fig)
            written += 1

        total_written += written
        print(f"  {written} written, {len(zoom_tiles)-written} empty")

    print(f"  [Tiles] Done — {total_written} tiles → {tile_root}")
    return tile_root