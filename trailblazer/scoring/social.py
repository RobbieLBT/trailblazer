"""
trailblazer/scoring/social.py — Community noise / social impact scoring (Phase 4)

Implements the social_impact(edge) term for edge_cost() in pathfinder.py.

    social_impact(edge) = (
        L_ref - 20 * log10(alt_m)      # L_ref ≈ 85 dB(A) at 1m for quadrotor
        * pop_density_per_km2           # LandScan raster, buffered 500m along edge
        * duration_s                    # edge length / cruise speed
        * time_of_day_factor            # +10 dB penalty 2200–0700 per DNL definition
        * sensitive_site_multiplier     # 3× within 500m of school/hospital (OSM)
    )

Voltage discount on social_impact (applied in pathfinder.edge_cost):
    765 kV → 0.10×    500 kV → 0.20×    345 kV → 0.40×
    230 kV → 0.65×    115 kV → 0.85×    cell/off-ROW → 1.00×

Phase 4 build order:
  1. Register at landscan.ornl.gov and download LandScan USA → data/population/
  2. Open with rasterio; build a small query wrapper that returns
     population-density-per-km² for a bounding box
  3. Edge buffer: for a (lat1,lon1)→(lat2,lon2) segment, create a
     500m-buffered corridor polygon and sum raster cells within it
  4. Wire into edge_cost(): pop_along_edge * noise_model * time_of_day
  5. Streamlit slider controls λ2 (noise_weight) in real time
"""

from __future__ import annotations

from datetime import datetime


# Voltage class → social impact multiplier (transmission ROW discount)
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
    Selects the closest voltage tier (rounds down).
    """
    tiers = sorted(VOLTAGE_SOCIAL_MULTIPLIER.keys(), reverse=True)
    for tier in tiers:
        if voltage_kv >= tier:
            return VOLTAGE_SOCIAL_MULTIPLIER[tier]
    return 1.00


def social_impact(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    cruise_kts: float,
    voltage_kv: float = 0.0,
    departure_time: datetime | None = None,
    pop_raster=None,          # rasterio DatasetReader (Phase 4)
) -> float:
    """
    Return a social-impact score (dimensionless, normalised to equivalent
    flight-minutes) for one graph edge.

    Phase 1–3: returns 0.0 (stub).
    Phase 4: replace with full population-weighted noise model.
    """
    # Phase 4: uncomment and implement
    # pop_density = _query_population(pop_raster, lat1, lon1, lat2, lon2)
    # noise_db = 85.0 - 20 * math.log10(max(alt_m, 1.0))
    # duration_s = _edge_nm(lat1, lon1, lat2, lon2) / cruise_kts * 3600
    # tod_factor = 3.162 if _is_night(departure_time) else 1.0  # +10 dB DNL
    # raw = noise_db * pop_density * duration_s * tod_factor
    # return raw * voltage_multiplier(voltage_kv)
    return 0.0
