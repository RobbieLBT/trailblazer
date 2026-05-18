"""
trailblazer/elevation.py — Terrain elevation fetch for graph nodes

Populates Fix.elevation_m for all nodes in a GraphData using the
Open-Topo-Data API (SRTM 30m, free, no API key required).

    https://api.opentopodata.org/v1/srtm30m

Usage
─────
    from trailblazer.elevation import fetch_elevations
    fetch_elevations(graph_data)   # modifies Fix.elevation_m in-place

Called from build_graph.py when --elevation flag is passed.

Disk cache
──────────
Elevation values are expensive to re-fetch (100 locations/request, ~1 req/s).
A flat JSON cache at data/elevation_cache.json stores {coord_key: elevation_m}
entries keyed on "lat,lon" strings rounded to 5 decimal places (~1m precision).

On each call:
  1. Any fix whose coordinate is already in the cache is populated immediately
     (no API call, no delay).
  2. Only fixes not in the cache are batched and sent to the API.
  3. Newly fetched values are written back to the cache.

This means the second and subsequent builds for the same AO are instant:
a ~800-node transmission graph that took 10s the first time takes <1s thereafter.
The cache grows as new bboxes are added; no entries are ever invalidated
(terrain elevation doesn't change).

Rate limits
───────────
Open-Topo-Data: 100 locations per request, ~1 req/sec for the public instance.
For a typical transmission graph (~800 nodes): ~10s first build, <1s cached.
For the cell tower graph (~300-1800 nodes depending on grid_cell_km): similar.

SRTM 30m coverage
─────────────────
Full continental US coverage.  Voids (water bodies, rare gaps) return None;
those nodes fall back to 0m in the altitude penalty calculation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import requests

from .types import Fix, GraphData

API_URL          = "https://api.opentopodata.org/v1/srtm30m"
BATCH_SIZE       = 100                              # API hard limit per request
MIN_DELAY        = 1.1                              # seconds between requests
DEFAULT_CACHE    = Path("data/elevation_cache.json")
_COORD_PRECISION = 5                                # decimal places for cache key (~1m)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _coord_key(lat: float, lon: float) -> str:
    """Stable dict key for a coordinate, rounded to _COORD_PRECISION decimals."""
    return f"{lat:.{_COORD_PRECISION}f},{lon:.{_COORD_PRECISION}f}"


def _load_cache(cache_path: Path) -> dict[str, float]:
    """Load the elevation cache from disk.  Returns {} on missing file or error."""
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [Elevation] Warning: could not read cache ({cache_path}): {exc}")
    return {}


def _save_cache(cache: dict[str, float], cache_path: Path) -> None:
    """Write the elevation cache to disk.  Silent on failure — non-fatal."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # separators=(",",":") → compact JSON; ~20 bytes per entry
        cache_path.write_text(
            json.dumps(cache, separators=(",", ":")), encoding="utf-8"
        )
    except Exception as exc:
        print(f"  [Elevation] Warning: could not save cache ({cache_path}): {exc}")


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_elevations(
    graph_data: GraphData,
    batch_size:  int = BATCH_SIZE,
    verbose:     bool = True,
    cache_path:  str | Path | None = DEFAULT_CACHE,
) -> int:
    """
    Fetch terrain elevation for every Fix in graph_data and store in Fix.elevation_m.

    Nodes already having elevation_m set are skipped.  Nodes whose coordinate
    is in the disk cache are populated without any API call.  Only remaining
    nodes hit the network.

    Parameters
    ----------
    graph_data  : GraphData whose Fix.elevation_m values are populated in-place.
    batch_size  : Locations per API request (max 100).
    verbose     : Print progress messages.
    cache_path  : Path to the JSON elevation cache.  Pass None to disable caching.

    Returns
    -------
    Number of nodes whose elevation_m was set (from cache or API) this call.
    """
    _cache_path = Path(cache_path) if cache_path else None

    # ── Load cache ────────────────────────────────────────────────────────
    cache: dict[str, float] = _load_cache(_cache_path) if _cache_path else {}

    # ── Pre-populate from cache ───────────────────────────────────────────
    from_cache  = 0
    pending: list[Fix] = []

    for fix in graph_data.fixes.values():
        if fix.elevation_m is not None:
            continue                              # already set (e.g. prior pkl)
        key = _coord_key(fix.lat, fix.lon)
        if key in cache:
            fix.elevation_m = cache[key]
            from_cache += 1
        else:
            pending.append(fix)

    if verbose and from_cache:
        print(f"  [Elevation] {from_cache:,} node(s) loaded from cache "
              f"({_cache_path or 'in-memory'})")

    if not pending:
        if verbose:
            total = sum(1 for f in graph_data.fixes.values() if f.elevation_m is not None)
            print(f"  [Elevation] All {total:,} nodes have elevation data (no fetch needed).")
        return from_cache

    # ── Fetch from API ────────────────────────────────────────────────────
    if verbose:
        n_req = math.ceil(len(pending) / batch_size)
        est_s = n_req * MIN_DELAY
        print(f"  [Elevation] Fetching {len(pending):,} new node(s) "
              f"({n_req} request(s), ~{est_s:.0f}s)…")

    from_api = 0
    for i in range(0, len(pending), batch_size):
        batch     = pending[i : i + batch_size]
        locations = "|".join(f"{f.lat:.6f},{f.lon:.6f}" for f in batch)

        try:
            resp = requests.post(
                API_URL,
                json={"locations": locations},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            for fix, result in zip(batch, results):
                elev = result.get("elevation")
                fix.elevation_m = float(elev) if elev is not None else 0.0
                # Write to cache regardless of 0.0 (avoids re-fetching water voids)
                cache[_coord_key(fix.lat, fix.lon)] = fix.elevation_m
                from_api += 1

        except Exception as exc:
            if verbose:
                print(f"  [Elevation] Batch {i // batch_size + 1} failed: {exc} — "
                      f"setting affected nodes to 0m")
            for fix in batch:
                if fix.elevation_m is None:
                    fix.elevation_m = 0.0
                    cache[_coord_key(fix.lat, fix.lon)] = 0.0

        if verbose and len(pending) > batch_size:
            done = min(i + batch_size, len(pending))
            print(f"  [Elevation] {done:,}/{len(pending):,} fetched…", end="\r")

        if i + batch_size < len(pending):
            time.sleep(MIN_DELAY)

    if verbose:
        print(f"  [Elevation] Done. "
              f"{from_cache:,} from cache + {from_api:,} fetched = "
              f"{from_cache + from_api:,} total nodes with elevation.")

    # ── Save updated cache ────────────────────────────────────────────────
    if _cache_path and from_api > 0:
        _save_cache(cache, _cache_path)
        if verbose:
            print(f"  [Elevation] Cache updated → {_cache_path} "
                  f"({len(cache):,} entries)")

    return from_cache + from_api


# ── Terrain profile sampler ───────────────────────────────────────────────────

def elevation_profile(
    fix_a: Fix,
    fix_b: Fix,
    n_samples: int = 5,
) -> Optional[list[float]]:
    """
    Sample terrain elevation along the great-circle path between two fixes.
    Returns a list of n_samples elevation values (metres), or None on failure.

    Used for the in-app terrain profile chart.  Does not use the node cache
    (interpolated points between nodes are not permanent graph nodes).
    """
    if n_samples < 2:
        return None

    lats = [fix_a.lat + i / (n_samples - 1) * (fix_b.lat - fix_a.lat)
            for i in range(n_samples)]
    lons = [fix_a.lon + i / (n_samples - 1) * (fix_b.lon - fix_a.lon)
            for i in range(n_samples)]

    locations = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in zip(lats, lons))
    try:
        resp = requests.post(API_URL, json={"locations": locations}, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [float(r.get("elevation") or 0) for r in results]
    except Exception:
        return None