"""
trailblazer/weather/gairmet.py — G-AIRMET polygon weather provider

Fetches G-AIRMETs from the Aviation Weather Center API and exposes them as
a WeatherProvider.  G-AIRMETs are polygon-based, time-windowed advisories
covering IFR, MTN OBSCN, turbulence, icing, etc.

AWC G-AIRMET API:
  https://aviationweather.gov/api/data/gairmet
  Returns JSON array.

AWC JSON API response format notes (confirmed from live data)
─────────────────────────────────────────────────────────────
• Do NOT send a `hazard=` filter parameter — comma-separated values return
  400 since early 2026.  Fetch all and filter client-side.
• `geom` is a JSON-encoded *string*, not a dict:
    "geom": "{\"type\":\"Polygon\",\"coordinates\":[[...]]}"
  Must be parsed with json.loads() before calling .get().
• Time fields are Unix timestamps (`validTimeFrom`, `validTimeTo`), not
  ISO strings.  The helpers below handle both formats for robustness.

Flight category mapping:
  IFR       → IFR
  MTN OBSCN → MVFR
  Other hazards → no flight-category impact (TURB, ICE, etc.)
"""

from __future__ import annotations

import json as _json
from datetime import datetime, timezone
from typing import Optional

import requests


_FC_RANK = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3, "UNKNOWN": 4}

_HAZARD_TO_FC = {
    "IFR":       "IFR",
    "MTN OBSCN": "MVFR",
}

# Hazards we care about — all others silently ignored
_RELEVANT_HAZARDS = frozenset({"IFR", "MTN OBSCN"})

AWC_GAIRMET_URL = "https://aviationweather.gov/api/data/gairmet"


class GAirmetProvider:
    """
    Weather provider backed by AWC G-AIRMET polygons.

    Fetches and caches the current G-AIRMET dataset on first call.
    flight_category() returns the worst IFR/MVFR condition at (lat, lon, t)
    from any overlapping polygon valid at time t.

    Cache is per-instance; invalidated after cache_minutes.
    """

    def __init__(self, cache_minutes: int = 60) -> None:
        self._advisories: list[dict] = []
        self._loaded_at: Optional[datetime] = None
        self._cache_minutes = cache_minutes

    def refresh(self) -> None:
        """Fetch current G-AIRMETs from AWC and cache them."""
        try:
            resp = requests.get(
                AWC_GAIRMET_URL,
                # No hazard= filter — AWC rejects comma-separated values with
                # 400 since early 2026.  Filter client-side instead.
                params={"format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json() or []
            # Client-side filter: keep only relevant hazard types
            self._advisories = [
                a for a in raw
                if (a.get("hazard") or "").upper().strip() in _RELEVANT_HAZARDS
            ]
            self._loaded_at = datetime.now(tz=timezone.utc)
            print(f"  [G-AIRMET] {len(raw)} advisories fetched, "
                  f"{len(self._advisories)} IFR/MTN after filter")
        except Exception as exc:
            print(f"  [G-AIRMET] Fetch failed: {exc} — G-AIRMET layer disabled")
            self._advisories = []
            self._loaded_at = datetime.now(tz=timezone.utc)

    def _ensure_loaded(self) -> None:
        if self._loaded_at is None:
            self.refresh()
            return
        age = (datetime.now(tz=timezone.utc) - self._loaded_at).total_seconds() / 60
        if age > self._cache_minutes:
            self.refresh()

    def flight_category(self, lat: float, lon: float, t: datetime) -> str:
        """Return worst G-AIRMET-based flight category at (lat, lon) at time t."""
        self._ensure_loaded()
        if not self._advisories:
            return "VFR"

        worst = "VFR"
        t_utc = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)

        for advisory in self._advisories:
            fc = _hazard_to_category(advisory)
            if fc == "VFR":
                continue
            if not _is_valid_at(advisory, t_utc):
                continue
            if not _point_in_advisory(lat, lon, advisory):
                continue
            if _FC_RANK.get(fc, 0) > _FC_RANK.get(worst, 0):
                worst = fc
            if worst == "IFR":
                break

        return worst


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hazard_to_category(advisory: dict) -> str:
    hazard = (advisory.get("hazard") or "").upper().strip()
    return _HAZARD_TO_FC.get(hazard, "VFR")


def _is_valid_at(advisory: dict, t: datetime) -> bool:
    """
    Check whether advisory is valid at time t.

    Handles two formats:
      • AWC JSON API live response: Unix timestamps in validTimeFrom / validTimeTo
      • Older / alternative format:  ISO strings in validTime / endTime
    """
    try:
        # ── AWC JSON API format: Unix timestamps ──────────────────────────
        from_ts = advisory.get("validTimeFrom")
        to_ts   = advisory.get("validTimeTo")
        if from_ts is not None or to_ts is not None:
            if from_ts is not None:
                from_dt = datetime.fromtimestamp(int(from_ts), tz=timezone.utc)
                if t < from_dt:
                    return False
            if to_ts is not None:
                to_dt = datetime.fromtimestamp(int(to_ts), tz=timezone.utc)
                if t > to_dt:
                    return False
            return True

        # ── Fallback: ISO string fields ───────────────────────────────────
        valid_str = advisory.get("validTime") or advisory.get("valid_time") or ""
        end_str   = advisory.get("endTime")   or advisory.get("end_time")   or ""
        if not valid_str:
            return True   # no time info → treat as always valid
        valid_dt = _parse_iso(valid_str)
        if valid_dt and t < valid_dt:
            return False
        if end_str:
            end_dt = _parse_iso(end_str)
            if end_dt and t > end_dt:
                return False
        return True
    except Exception:
        return True   # malformed time fields → don't exclude


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_geom(raw) -> Optional[dict]:
    """
    Normalise a geometry value to a dict, or return None.

    AWC JSON API returns geom as a JSON-encoded *string*:
        "geom": "{\"type\":\"Polygon\",\"coordinates\":[[...]]}"
    This is distinct from a true GeoJSON geometry object.  Parse it if needed.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return None


def _point_in_advisory(lat: float, lon: float, advisory: dict) -> bool:
    """Test whether (lat, lon) falls inside the advisory's polygon(s)."""
    # geom / geometry: may be a dict (old format) or a JSON-encoded string
    # (AWC live JSON API).  _parse_geom handles both.
    geom = _parse_geom(advisory.get("geom") or advisory.get("geometry"))
    if geom is not None:
        gtype  = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            return any(_ray_cast(lat, lon, ring) for ring in coords)
        elif gtype == "MultiPolygon":
            for polygon in coords:
                if any(_ray_cast(lat, lon, ring) for ring in polygon):
                    return True
            return False

    # Fallback: raw coord list (some older API responses)
    raw = advisory.get("coords") or advisory.get("points") or []
    if raw:
        return _ray_cast(lat, lon, _normalize_ring(raw))

    return False


def _normalize_ring(ring: list) -> list:
    """
    Coerce a ring to [[lon, lat], ...] with float values.

    AWC's non-GeoJSON fields (coords / points) have been observed as:
      - [[lon, lat], ...]                     — list of lists (may be str)
      - [{"lon": x, "lat": y}, …]             — dict form
      - [{"longitude": x, "latitude": y}, …]  — verbose dict form
    All coordinate values are cast to float to guard against string numerics.
    """
    if not ring:
        return ring
    first = ring[0]
    if isinstance(first, dict):
        def _extract(pt: dict) -> list:
            if "lon" in pt and "lat" in pt:
                return [float(pt["lon"]), float(pt["lat"])]
            if "longitude" in pt and "latitude" in pt:
                return [float(pt["longitude"]), float(pt["latitude"])]
            vals = list(pt.values())
            return [float(vals[0]), float(vals[1])] if len(vals) >= 2 else [0.0, 0.0]
        return [_extract(pt) for pt in ring]
    # Sequence of sequences — ensure values are float
    return [[float(c) for c in pt] for pt in ring]


def _ray_cast(lat: float, lon: float, ring: list) -> bool:
    """Ray-casting point-in-polygon.  ring: [[lon, lat], ...] GeoJSON convention."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside