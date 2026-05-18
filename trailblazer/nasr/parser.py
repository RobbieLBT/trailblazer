"""
trailblazer/nasr/parser.py — FAA NASR Subscription file parser

Parses NAV.txt, AWY.txt, and FIX.txt from the FAA 28-day NASR cycle.
Returns GraphData — the same type used by tx_parser — so the routing
layer handles both graph sources identically.

Download: https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
Extract from the zip into data/nasr/:  NAV.txt  AWY.txt  FIX.txt

Column offsets are based on the published NASR specification (cycle 2024+).
If a parse produces unexpected results, compare against the Layout/ directory
included in the NASR zip.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..types import Fix, AirwaySegment, GraphData


# ── DMS parser ─────────────────────────────────────────────────────────────────

_DMS_PATTERN = re.compile(
    r"""
    (\d{2,3})       # degrees
    [°\-\s]?
    (\d{2})         # minutes
    ['\-\s]?
    ([\d.]+)        # seconds (may include decimal)
    ["\s]?
    ([NSEW])        # hemisphere
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _parse_dms(raw: str) -> Optional[float]:
    raw = raw.strip()
    if not raw:
        return None
    m = _DMS_PATTERN.search(raw)
    if not m:
        try:
            return float(raw.rstrip("NnSsEeWw"))
        except ValueError:
            return None
    deg, mn, sec, hemi = m.groups()
    val = float(deg) + float(mn) / 60.0 + float(sec) / 3600.0
    if hemi.upper() in ("S", "W"):
        val = -val
    return val


# ── NAV.txt parser ─────────────────────────────────────────────────────────────

_NAV1_REC_TYPE = slice(0, 4)
_NAV1_IDENT    = slice(4, 9)
_NAV1_TYPE     = slice(9, 29)
_NAV1_NAME     = slice(29, 79)
_NAV1_STATE    = slice(79, 81)
_NAV1_LAT_DMS  = slice(371, 385)
_NAV1_LON_DMS  = slice(385, 400)


def _parse_nav_type(raw_type: str) -> str:
    t = raw_type.strip().upper()
    for canonical in ("VORTAC", "VOR/DME", "VOR-DME", "VORDME", "VOR", "NDB", "TACAN", "DME"):
        if canonical in t:
            return canonical
    return t[:10]


def parse_nav(nav_path: Path) -> dict[str, Fix]:
    """Parse NAV.txt → VOR/VORTAC/VOR-DME Fix objects."""
    fixes: dict[str, Fix] = {}
    skipped = errors = 0

    with open(nav_path, encoding="latin-1") as f:
        for line in f:
            if len(line) < 401:
                continue
            if line[_NAV1_REC_TYPE] != "NAV1":
                continue

            ident    = line[_NAV1_IDENT].strip()
            fix_type = _parse_nav_type(line[_NAV1_TYPE])
            name     = line[_NAV1_NAME].strip().title()
            state    = line[_NAV1_STATE].strip()
            lat      = _parse_dms(line[_NAV1_LAT_DMS])
            lon      = _parse_dms(line[_NAV1_LON_DMS])

            if not any(t in fix_type for t in ("VOR", "VORTAC", "DME")):
                skipped += 1
                continue
            if lat is None or lon is None or not ident:
                errors += 1
                continue

            fixes[ident] = Fix(
                ident=ident, lat=lat, lon=lon,
                fix_type=fix_type, name=name, state=state,
            )

    print(f"  [NAV]  {len(fixes):,} VOR/VORTAC fixes loaded "
          f"({skipped} non-VOR skipped, {errors} parse errors)")
    return fixes


# ── FIX.txt parser ─────────────────────────────────────────────────────────────

_FIX1_REC_TYPE = slice(0, 4)
_FIX1_IDENT    = slice(4, 34)
_FIX1_STATE    = slice(34, 36)
_FIX1_LAT_DMS  = slice(66, 80)
_FIX1_LON_DMS  = slice(80, 95)


def parse_fix(fix_path: Path) -> dict[str, Fix]:
    """Parse FIX.txt → named intersection Fix objects."""
    fixes: dict[str, Fix] = {}
    errors = 0

    with open(fix_path, encoding="latin-1") as f:
        for line in f:
            if len(line) < 96:
                continue
            if line[_FIX1_REC_TYPE] != "FIX1":
                continue

            ident = line[_FIX1_IDENT].strip()
            state = line[_FIX1_STATE].strip()
            lat   = _parse_dms(line[_FIX1_LAT_DMS])
            lon   = _parse_dms(line[_FIX1_LON_DMS])

            if lat is None or lon is None or not ident:
                errors += 1
                continue

            fixes[ident] = Fix(
                ident=ident, lat=lat, lon=lon,
                fix_type="INT", state=state,
            )

    print(f"  [FIX]  {len(fixes):,} intersections loaded ({errors} parse errors)")
    return fixes


# ── AWY.txt parser ─────────────────────────────────────────────────────────────

_AWY_REC_TYPE   = slice(0, 4)
_AWY1_AIRWAY_ID = slice(4, 9)
_AWY1_TYPE      = slice(9, 10)
_AWY2_AIRWAY_ID = slice(4, 9)
_AWY2_SEQ       = slice(9, 14)
_AWY2_FIX_IDENT = slice(14, 34)


def parse_awy(
    awy_path: Path,
    airway_filter: Optional[set[str]] = None,
    type_filter: str = "L",
) -> list[AirwaySegment]:
    """
    Parse AWY.txt → AirwaySegment pairs.

    airway_filter : restrict to specific airway IDs (e.g. {"V268", "V20"})
    type_filter   : "L" = Victor/low, "H" = jet/high, None = all
    """
    airways: dict[str, list[tuple[int, str]]] = {}
    airway_types: dict[str, str] = {}

    with open(awy_path, encoding="latin-1") as f:
        for line in f:
            if len(line) < 10:
                continue
            rec_type = line[_AWY_REC_TYPE]

            if rec_type == "AWY1":
                awy_id = line[_AWY1_AIRWAY_ID].strip()
                airway_types[awy_id] = line[_AWY1_TYPE].strip()

            elif rec_type == "AWY2":
                awy_id = line[_AWY2_AIRWAY_ID].strip()
                if type_filter and airway_types.get(awy_id, "") != type_filter:
                    continue
                if airway_filter and awy_id not in airway_filter:
                    continue
                try:
                    seq = int(line[_AWY2_SEQ].strip())
                except ValueError:
                    continue
                fix_ident = line[_AWY2_FIX_IDENT].strip()
                if fix_ident:
                    airways.setdefault(awy_id, []).append((seq, fix_ident))

    segments: list[AirwaySegment] = []
    for awy_id, points in airways.items():
        points.sort(key=lambda x: x[0])
        for i in range(len(points) - 1):
            seq_a, fix_a = points[i]
            seq_b, fix_b = points[i + 1]
            segments.append(AirwaySegment(
                airway_id=awy_id,
                from_ident=fix_a,
                to_ident=fix_b,
                seq_from=seq_a,
                seq_to=seq_b,
                voltage_kv=0.0,   # NASR airways are off-ROW
            ))

    print(f"  [AWY]  {len(airways)} airways → {len(segments):,} segments loaded")
    return segments


# ── Combined loader ────────────────────────────────────────────────────────────

def load_nasr(
    nasr_dir: str | Path,
    airway_filter: Optional[set[str]] = None,
    include_intersections: bool = True,
) -> GraphData:
    """
    Load all NASR data from a directory containing NAV.txt, AWY.txt, FIX.txt.

    Returns GraphData with source="nasr".
    VOR entries take precedence on ident collision with FIX.txt intersections.
    """
    d = Path(nasr_dir)
    nav_path = d / "NAV.txt"
    awy_path = d / "AWY.txt"
    fix_path = d / "FIX.txt"

    for p in (nav_path, awy_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required NASR file not found: {p}\n"
                f"Download from: https://www.faa.gov/air_traffic/flight_info/"
                f"aeronav/aero_data/NASR_Subscription/"
            )

    print(f"\n[NASR] Loading from {d}")

    fixes = parse_nav(nav_path)

    if include_intersections and fix_path.exists():
        int_fixes = parse_fix(fix_path)
        added = sum(1 for ident, fix in int_fixes.items()
                    if ident not in fixes and not fixes.update({ident: fix}))
        print(f"  [FIX]  {added} intersections merged (non-duplicates)")
    elif include_intersections:
        print("  [FIX]  FIX.txt not found — intersections skipped")

    segments = parse_awy(awy_path, airway_filter=airway_filter)

    all_idents = {s.from_ident for s in segments} | {s.to_ident for s in segments}
    missing = all_idents - set(fixes.keys())
    if missing:
        sample = sorted(missing)[:10]
        print(f"  [WARN] {len(missing)} fix idents in AWY.txt have no position data "
              f"(will be dropped): {sample}{'...' if len(missing) > 10 else ''}")

    print(f"  [NASR] {len(fixes):,} total fixes, {len(segments):,} segments\n")
    return GraphData(fixes=fixes, segments=segments, source="nasr")
