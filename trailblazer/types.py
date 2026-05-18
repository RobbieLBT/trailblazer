"""
trailblazer/types.py — Shared graph dataclasses

These types form the contract between every data-source parser
(nasr/parser.py, infra/tx_parser.py, infra/tower_parser.py) and the
routing layer (routing/pathfinder.py).

All parsers return a GraphData object. The pathfinder and export modules
consume GraphData and never import parser modules directly.

Fix
    A navigable waypoint: VOR, named intersection, EIA substation,
    FCC cell tower, or any other point that can appear as a graph node.

AirwaySegment
    A directed edge between two Fix idents: a Victor airway segment,
    a transmission line segment, or a tower-mesh Delaunay edge.
    voltage_kv is non-zero for transmission segments and drives the
    social-impact discount in edge_cost() (Phase 4).

GraphData
    The full graph dataset handed from a parser to build_graph() /
    the CLI. Equivalent to the old NASRData; renamed to reflect that
    the graph source is no longer exclusively NASR.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fix:
    """A navigable waypoint (graph node)."""
    ident: str
    lat: float          # decimal degrees N  (WGS84)
    lon: float          # decimal degrees E  (negative = W)
    fix_type: str       # "VOR" | "VORTAC" | "INT" | "SUBSTATION" | "TOWER" | ...
    name: str       = ""
    state: str      = ""
    elevation_m: float | None = None   # terrain elevation AGL from DEM; None = not yet fetched


@dataclass
class AirwaySegment:
    """A directed edge between two Fix idents (graph edge)."""
    airway_id: str      # e.g. "V268", "500KV", "TOWER_MESH"
    from_ident: str
    to_ident: str
    seq_from: int       # position of from_fix in source sequence
    seq_to: int         # position of to_fix   in source sequence
    voltage_kv: float = 0.0
    # 0 = off-ROW / cell mesh; non-zero = transmission ROW
    # Used by social_impact() voltage discount table in Phase 4.


@dataclass
class GraphData:
    """
    Complete graph dataset produced by any parser.

    fixes    : {ident → Fix}  — all nodes that may appear in segments
    segments : ordered list of directed edges
    source   : short label for the data origin, e.g. "nasr", "transmission"
    """
    fixes: dict[str, Fix]
    segments: list[AirwaySegment]
    source: str = "unknown"