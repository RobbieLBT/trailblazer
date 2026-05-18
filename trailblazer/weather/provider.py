"""
trailblazer/weather/provider.py — WeatherProvider protocol and implementations

Defines the interface contract between the route planner and any weather data
source.  The pathfinder only depends on this protocol — it never imports
Weatherboy or any other specific weather module directly.

Implementations
───────────────
  WeatherboyProvider  wraps Weatherboy's interpolated obs field (recommended)
  GAirmetProvider     queries AWC G-AIRMETs as polygon-based forecast layer
  CompositeProvider   worst-of from multiple providers (stack both above)
  MockProvider        controllable wx per fix, for testing and demos

WeatherboyProvider — actual API notes
──────────────────────────────────────
Weatherboy has no Interpolator class.  The interface is three functions in
weather/interpolate.py:

    interpolate_obs_at_time(obs_data, t)        → {stn: obs_dict} snapshot
    query_point(snapshot, lat, lon, method, κ)  → field dict
        field keys: U, V, speed, wdir, temp, pressure_hpa, vsby, ceiling_ft

Flight category is derived from ceiling_ft + vsby — no flight_category field
comes directly from a Weatherboy function; this module computes it from the
field dict using the same thresholds as mission/traverse.py.

WeatherboyProvider.__init__ therefore takes obs_data (the dict returned by
weather.fetch.fetch_metars), not an interpolator object.  cli.py and
build_graph.py are responsible for fetching obs_data from the config XML
before constructing the provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from .gairmet import GAirmetProvider   # noqa: F401 — re-exported for convenience


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class WeatherProvider(Protocol):
    """
    Minimal interface between the route planner and any weather data source.

    The pathfinder calls flight_category() once per graph edge midpoint at
    the estimated arrival time at that point, enabling time-accurate weather
    sampling along a multi-hour route.
    """

    def flight_category(self, lat: float, lon: float, t: datetime) -> str:
        """
        Return the forecast or observed flight category at (lat, lon) at time t.
        Returns one of: "VFR" | "MVFR" | "IFR" | "LIFR" | "UNKNOWN"
        lat, lon : decimal degrees (WGS84); lon negative west of prime meridian.
        t        : UTC datetime of query (estimated arrival time at this point).
        """
        ...


# ── Flight category utilities ─────────────────────────────────────────────────

FC_RANK: dict[str, int] = {
    "VFR":     0,
    "MVFR":    1,
    "IFR":     2,
    "LIFR":    3,
    "UNKNOWN": 4,
}


def category_from_obs(ceiling_ft: float | None, vis_sm: float | None) -> str:
    """Derive flight category from ceiling and visibility per FAA definitions."""
    cat = "VFR"
    if ceiling_ft is not None:
        if ceiling_ft <= 500:
            cat = "LIFR"
        elif ceiling_ft <= 1000:
            cat = "IFR"
        elif ceiling_ft <= 3000:
            cat = "MVFR"

    if vis_sm is not None:
        vis_cat = "VFR"
        if vis_sm < 1.0:
            vis_cat = "LIFR"
        elif vis_sm < 3.0:
            vis_cat = "IFR"
        elif vis_sm <= 5.0:
            vis_cat = "MVFR"
        if FC_RANK[vis_cat] > FC_RANK[cat]:
            cat = vis_cat

    return cat


def worst_category(categories: list[str]) -> str:
    """Return the worst (highest-rank) category from a list."""
    if not categories:
        return "UNKNOWN"
    return max(categories, key=lambda c: FC_RANK.get(c.upper(), 4))


# ── WeatherboyProvider ────────────────────────────────────────────────────────

class WeatherboyProvider:
    """
    Adapter wrapping Weatherboy's interpolated obs field.

    Takes obs_data from weather.fetch.fetch_metars and uses Weatherboy's
    interpolate_obs_at_time + query_point functions to evaluate conditions
    at any (lat, lon, t).

    Usage
    ─────
    # 1. Add weatherboy/ to sys.path (or pip install -e ../weatherboy once
    #    pyproject.toml exists)
    # 2. Fetch observations for your AO and time window:
    #
    #    import sys; sys.path.insert(0, "../weatherboy")
    #    from weather.fetch import fetch_metars
    #    obs_data = fetch_metars(["KORF", "KCHO", "KLYH", "KROA", "KSHD"],
    #                            start=departure_time - timedelta(hours=1),
    #                            end=departure_time + timedelta(hours=4))
    #    provider = WeatherboyProvider(obs_data)
    #
    # 3. Optionally wrap with GAirmetProvider in a CompositeProvider.

    The length scale κ is computed once from the first snapshot and reused
    across all calls — station positions don't change mid-mission.
    """

    def __init__(self, obs_data: dict, method: str = "barnes") -> None:
        """
        obs_data : {station_id: [obs_dict, ...]}  from weather.fetch.fetch_metars
        method   : 'barnes' | 'cressman'
        """
        self._obs_data = obs_data
        self._method   = method
        self._length_scale: float | None = None   # lazy-init on first call

    def _ensure_length_scale(self) -> None:
        if self._length_scale is not None:
            return
        try:
            # Import lazily so Trailblazer runs without Weatherboy installed;
            # only WeatherboyProvider.__init__ callers need it present.
            from weather.interpolate import interpolate_obs_at_time, compute_length_scale  # noqa
            first_t = min(
                o["time"]
                for obs_list in self._obs_data.values()
                for o in obs_list
            )
            snap = interpolate_obs_at_time(self._obs_data, first_t)
            self._length_scale = compute_length_scale(snap, self._method)
        except Exception:
            self._length_scale = 1.0   # safe fallback

    def flight_category(self, lat: float, lon: float, t: datetime) -> str:
        try:
            self._ensure_length_scale()
            from weather.interpolate import interpolate_obs_at_time, query_point  # noqa

            snapshot = interpolate_obs_at_time(self._obs_data, t)
            if not snapshot:
                return "UNKNOWN"

            fld = query_point(snapshot, lat, lon, self._method, self._length_scale)
            if not fld:
                return "UNKNOWN"

            # Derive flight category from ceiling + visibility — same thresholds
            # as mission/traverse.py _flight_category()
            return category_from_obs(fld.get("ceiling_ft"), fld.get("vsby"))

        except Exception:
            return "UNKNOWN"


# ── CompositeProvider ─────────────────────────────────────────────────────────

class CompositeProvider:
    """
    Takes the worst-of flight category across multiple providers.

    Recommended production stack:
        CompositeProvider([
            WeatherboyProvider(interp),   # point observations, interpolated
            GAirmetProvider(),             # polygon-based forecast advisories
        ])
    """

    def __init__(self, providers: list[WeatherProvider]) -> None:
        self._providers = providers

    def flight_category(self, lat: float, lon: float, t: datetime) -> str:
        categories = [p.flight_category(lat, lon, t) for p in self._providers]
        return worst_category(categories)


# ── MockProvider ──────────────────────────────────────────────────────────────

@dataclass
class MockProvider:
    """
    Controllable weather provider for testing and portfolio demos.

    Assign per-fix conditions to simulate weather scenarios without
    network calls.  Falls back to `default` for any unregistered position.

    Usage:
        wx = MockProvider(default="VFR")
        wx.set("SHD", "IFR", lat=38.26, lon=-78.90)
        wx.set("HCH", "MVFR", lat=38.07, lon=-80.22)
    """

    default: str = "VFR"
    snap_radius_nm: float = 30.0

    _fix_wx: dict[str, str] = field(default_factory=dict, repr=False)
    _fix_positions: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)

    def set(self, ident: str, category: str, lat: float = 0.0, lon: float = 0.0) -> None:
        """Register a weather condition at a fix ident and optional position."""
        self._fix_wx[ident] = category.upper()
        if lat or lon:
            self._fix_positions[ident] = (lat, lon)

    def set_bulk(self, conditions: dict[str, str]) -> None:
        """Set multiple ident→category conditions at once."""
        for ident, cat in conditions.items():
            self.set(ident, cat)

    def flight_category(self, lat: float, lon: float, t: datetime) -> str:
        if not self._fix_positions:
            return self.default

        best_dist = float("inf")
        best_cat  = self.default
        for ident, (flat, flon) in self._fix_positions.items():
            d = _haversine_nm(lat, lon, flat, flon)
            if d < best_dist:
                best_dist = d
                best_cat  = self._fix_wx.get(ident, self.default)

        return best_cat if best_dist <= self.snap_radius_nm else self.default


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ, dλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
