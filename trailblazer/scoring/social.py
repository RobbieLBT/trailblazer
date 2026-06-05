"""
trailblazer/scoring/social.py — Community noise / social impact scoring

Implements the population impact factor for the noise_weight term in
apply_weights() / pathfinder.py.

Formula
───────
    noise_score(edge) = log1p(pop_sum / cruise_kts)   [then normalised to [0,1]]

where pop_sum is the sum of LandScan USA ambient population within a 250 m
ground buffer of the edge corridor.  Dividing by cruise_kts converts the
raw spatial count to a time-domain exposure proxy: slower = more person-
seconds of overflight = higher penalty.

Normalization is performed inside apply_weights() on every reweight pass
(slider move), using the cruise_kts active at that moment.  This keeps the
wire fully exposed for aircraft-design parameter sweeps without requiring a
graph rebuild.

Voltage discount (applied inside apply_weights via voltage_multiplier()):
    765 kV → 0.10×    500 kV → 0.20×    345 kV → 0.40×
    230 kV → 0.65×    115 kV → 0.85×    cell / off-ROW → 1.00×

Acoustic characterisation note (future work — see working_knowledge.md §12f)
──────────────────────────────────────────────────────────────────────────────
The current formulation is vehicle-agnostic.  A proper dose metric requires
NPD (Noise-Power-Distance) curves per vehicle type and should replace the
static 250 m buffer with a range-weighted population integral once acoustic
test data is available.  The edge attribute contract (pop_sum stored raw,
normalised at query time) is designed to accept this upgrade with no changes
to apply_weights() or the Streamlit UI.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ── Voltage discount table ────────────────────────────────────────────────────

VOLTAGE_SOCIAL_MULTIPLIER: dict[float, float] = {
    765: 0.10,
    500: 0.20,
    345: 0.40,
    230: 0.65,
    115: 0.85,
    0:   1.00,   # off-ROW / cell mesh
}


def voltage_multiplier(voltage_kv: float) -> float:
    """
    Return the social-impact multiplier for a given nominal voltage.
    Selects the closest voltage tier (rounds down to nearest tier).
    """
    tiers = sorted(VOLTAGE_SOCIAL_MULTIPLIER.keys(), reverse=True)
    for tier in tiers:
        if voltage_kv >= tier:
            return VOLTAGE_SOCIAL_MULTIPLIER[tier]
    return 1.00


# ── Stage-1 population scoring ────────────────────────────────────────────────

def compute_population_scores(
    G,
    landscan_path: str | Path,
    buffer_m: float = 250.0,
) -> None:
    """
    Sample LandScan USA ambient population raster along every graph edge.

    For each unique undirected edge pair (a, b), creates a buffer_m-wide
    corridor around the straight-line segment — projected to EPSG:3857 for
    metric accuracy — sums raster cells within it using
    rasterstats.zonal_stats(), and stores the raw count as edge['pop_sum']
    on both directed edges.  Edges with missing node coordinates silently
    receive pop_sum = 0.0.

    Also writes normalisation metadata to the graph for use in
    apply_weights():
        G.graph['pop_sum_min']  — min raw pop_sum across all edges
        G.graph['pop_sum_max']  — max raw pop_sum across all edges

    Parameters
    ----------
    G             : nx.DiGraph with node attrs lat, lon
    landscan_path : path to landscan-mosaic-unitedstates-v1.tif
                    (the main data band — NOT the colorized or CI variant)
    buffer_m      : ground buffer half-width in metres (default 250)

    Notes
    ─────
    - Requires: geopandas, shapely, rasterstats  (add rasterstats to deps)
    - LandScan USA ships in EPSG:4326; geometries are reprojected to
      EPSG:3857 for buffering then back to EPSG:4326 for raster sampling
    - Directed edges sharing the same geographic corridor receive equal pop_sum
    - Buffer radius of 250 m at 350 ft AGL is a deliberate lower bound;
      see working_knowledge.md §12b for rationale
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
        from rasterstats import zonal_stats
    except ImportError as exc:
        raise ImportError(
            f"Population scoring requires geopandas and rasterstats: {exc}\n"
            f"  pip install rasterstats"
        ) from exc

    landscan_path = Path(landscan_path)
    if not landscan_path.exists():
        raise FileNotFoundError(
            f"LandScan raster not found: {landscan_path}\n"
            f"Expected path:\n"
            f"  data/population/landscan-mosaic-unitedstates-v1-assets/"
            f"landscan-mosaic-unitedstates-v1.tif"
        )

    print(f"\n── Computing population scores ──")
    print(f"   raster : {landscan_path.name}")
    print(f"   buffer : {buffer_m:.0f} m")

    # ── Build unique undirected edge geometries ────────────────────────────
    # frozenset key ensures (a→b) and (b→a) share one geometry / one stats call
    seen:     dict[frozenset, tuple]         = {}   # key → (a, b, LineString | None)
    key_order: list[frozenset]               = []   # insertion order for result mapping

    for a, b in G.edges():
        key = frozenset((a, b))
        if key in seen:
            continue
        na   = G.nodes.get(a, {})
        nb   = G.nodes.get(b, {})
        lat_a, lon_a = na.get("lat"), na.get("lon")
        lat_b, lon_b = nb.get("lat"), nb.get("lon")

        if None in (lat_a, lon_a, lat_b, lon_b):
            seen[key] = (a, b, None)
        else:
            # GeoDataFrame convention: (x=lon, y=lat)
            seen[key] = (a, b, LineString([(lon_a, lat_a), (lon_b, lat_b)]))
        key_order.append(key)

    valid_keys  = [k for k in key_order if seen[k][2] is not None]
    valid_geoms = [seen[k][2] for k in valid_keys]
    valid_ab    = [(seen[k][0], seen[k][1]) for k in valid_keys]

    if not valid_geoms:
        print("  [Pop] No valid edge geometries — all pop_sum set to 0.0")
        _set_zero(G)
        return

    total_undirected = G.number_of_edges() // 2
    print(f"  [Pop] {len(valid_geoms):,} unique edges to sample "
          f"({total_undirected:,} total undirected)")

    # ── Project to EPSG:3857 → buffer → reproject to WGS84 ───────────────
    gdf = gpd.GeoDataFrame(
        {"edge_idx": range(len(valid_geoms))},
        geometry=valid_geoms,
        crs="EPSG:4326",
    )
    gdf_proj     = gdf.to_crs("EPSG:3857")
    gdf_buffered = gdf_proj.copy()
    gdf_buffered["geometry"] = gdf_proj.geometry.buffer(buffer_m)   # round caps
    gdf_wgs84 = gdf_buffered.to_crs("EPSG:4326")

    # ── One-pass zonal stats over all edge corridors ───────────────────────
    # all_touched=True: include raster cells that touch the polygon boundary —
    # important for thin corridors that might otherwise miss cells entirely
    print(f"  [Pop] Sampling raster (may take 30–120 s for large graphs)…")
    stats = zonal_stats(
        gdf_wgs84,
        str(landscan_path),
        stats=["sum"],
        nodata=0,
        all_touched=True,
        band=1,
    )
    print(f"  [Pop] Raster sampling complete.")

    # ── Map results back to directed edges ────────────────────────────────
    pop_sums: dict[frozenset, float] = {}
    for i, (a, b) in enumerate(valid_ab):
        raw = (stats[i] or {}).get("sum") or 0.0
        pop_sums[frozenset((a, b))] = max(0.0, float(raw))

    for a, b, data in G.edges(data=True):
        data["pop_sum"] = pop_sums.get(frozenset((a, b)), 0.0)

    # ── Store normalisation bounds in graph metadata ───────────────────────
    all_sums = [d.get("pop_sum", 0.0) for _, _, d in G.edges(data=True)]
    G.graph["pop_sum_min"] = float(np.min(all_sums))
    G.graph["pop_sum_max"] = float(np.max(all_sums))

    non_zero = sum(1 for v in all_sums if v > 0)
    print(
        f"  [Pop] pop_sum range: "
        f"{G.graph['pop_sum_min']:.0f} – {G.graph['pop_sum_max']:.0f}  |  "
        f"non-zero edges: {non_zero:,} / {len(all_sums):,}"
    )


def _set_zero(G) -> None:
    """Fallback: set pop_sum = 0.0 on all edges and zero out graph metadata."""
    for _, _, data in G.edges(data=True):
        data["pop_sum"] = 0.0
    G.graph["pop_sum_min"] = 0.0
    G.graph["pop_sum_max"] = 0.0


# ── Stage-1 population overlay PNG ───────────────────────────────────────────

def generate_population_overlay(
    G,
    landscan_path: str | Path,
    output_png: str | Path,
    output_meta: str | Path,
    target_width: int = 1000,
) -> None:
    """
    Generate a downsampled RGBA PNG of the LandScan raster clipped to the
    graph's bounding box, for use as a Folium ImageOverlay.

    Applies log1p normalisation + YlOrRd colormap; fully transparent where
    population = 0 so the basemap shows through.  The PNG is loaded by the
    Streamlit app as a base64 data URI — no static file server needed, which
    makes it safe for public Streamlit Cloud deployments.

    Parameters
    ----------
    G            : nx.DiGraph with node attrs lat, lon
    landscan_path: path to landscan-mosaic-unitedstates-v1.tif
    output_png   : destination path for the RGBA PNG (e.g. population_cell.png)
    output_meta  : destination path for the JSON bounds sidecar
    target_width : output image width in pixels (default 1000; keeps
                   base64 payload under ~400 KB for Folium)

    Requires: rasterio, numpy, matplotlib, Pillow
    """
    import json

    import matplotlib.cm as cm
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError("generate_population_overlay requires Pillow: pip install Pillow") from exc

    landscan_path = Path(landscan_path)
    output_png    = Path(output_png)
    output_meta   = Path(output_meta)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # ── Graph bounding box + margin ────────────────────────────────────────
    lats = [d.get("lat", 0.0) for _, d in G.nodes(data=True) if d.get("lat") is not None]
    lons = [d.get("lon", 0.0) for _, d in G.nodes(data=True) if d.get("lon") is not None]
    if not lats:
        print("  [PopOverlay] No node coordinates — skipping overlay generation.")
        return

    margin  = 0.25   # degrees of padding around graph extent
    min_lat = min(lats) - margin
    max_lat = max(lats) + margin
    min_lon = min(lons) - margin
    max_lon = max(lons) + margin

    # ── Read & resample raster ─────────────────────────────────────────────
    # Height is proportional to lat/lon extent so pixels are roughly square.
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    h_px     = max(1, round(target_width * lat_span / lon_span))

    print(f"\n── Generating population overlay PNG ──")
    print(f"   bbox   : lat [{min_lat:.2f}, {max_lat:.2f}]  "
          f"lon [{min_lon:.2f}, {max_lon:.2f}]")
    print(f"   size   : {target_width} × {h_px} px")

    with rasterio.open(landscan_path) as src:
        nodata  = src.nodata
        window  = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        data    = src.read(
            1,
            window=window,
            out_shape=(h_px, target_width),
            resampling=Resampling.average,
        ).astype(float)

    # ── Clean & transform ─────────────────────────────────────────────────
    # Zero out nodata (LandScan nodata is typically -9999 or similar)
    if nodata is not None:
        data[data == nodata] = 0.0
    data[data < 0] = 0.0

    data_log  = np.log1p(data)
    dmax      = data_log.max()
    data_norm = (data_log / dmax) if dmax > 0 else data_log

    # ── Apply colormap + alpha ─────────────────────────────────────────────
    # YlOrRd: yellow (sparse) → orange → red (dense).
    # Alpha is 0 for zero-pop cells so the basemap shows through cleanly.
    colormap  = cm.get_cmap("YlOrRd")
    rgba      = colormap(data_norm).astype(float)                 # (H, W, 4) [0,1]
    rgba[:, :, 3] = np.where(data > 0, 0.60, 0.0)                # 60% opacity where populated

    rgba_uint8 = (rgba * 255).clip(0, 255).astype(np.uint8)

    # rasterio reads rasters top-to-bottom (north first); Folium ImageOverlay
    # also expects north at top, so no flip needed.
    img = PILImage.fromarray(rgba_uint8, mode="RGBA")
    img.save(str(output_png), format="PNG", optimize=True)

    # ── Save bounds sidecar ───────────────────────────────────────────────
    meta = {
        "bounds":  [[min_lat, min_lon], [max_lat, max_lon]],
        "min_lat": min_lat, "max_lat": max_lat,
        "min_lon": min_lon, "max_lon": max_lon,
        "width_px": target_width, "height_px": h_px,
    }
    output_meta.write_text(json.dumps(meta, indent=2))

    size_kb = output_png.stat().st_size / 1024
    print(f"  [PopOverlay] Saved → {output_png.name} ({size_kb:.0f} KB)  "
          f"bounds sidecar → {output_meta.name}")


# ── Legacy per-edge stub (kept for API compatibility) ─────────────────────────

def social_impact(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    cruise_kts: float,
    voltage_kv: float = 0.0,
    departure_time=None,
    pop_raster=None,
) -> float:
    """
    Legacy per-edge social impact hook.  Returns 0.0.

    Phase 4 uses compute_population_scores() (Stage 1) + apply_weights()
    (Stage 2) instead of this per-edge function.  Retained for import
    compatibility with pathfinder.edge_cost() — that call path is now a
    no-op since noise_weight is handled entirely in apply_weights().
    """
    return 0.0
