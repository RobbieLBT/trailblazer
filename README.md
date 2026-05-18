# Trailblazer

**Intelligent pathfinding for BVLOS UAS operations**

Trailblazer is a two-stage Python framework that builds weighted routing graphs from real infrastructure and airspace data, then finds and visualises optimal BVLOS (Beyond Visual Line of Sight) drone routes through an interactive Streamlit app. It was developed by Commonwealth Drone Solutions.

---

## How it works

**Stage 1 — Graph build (`build_graph.py`)**

Ingests infrastructure, airspace, elevation, and weather data and serialises a weighted [NetworkX](https://networkx.org/) graph to a `.pkl` file. Each edge stores attributes including flight time, elevation gain, airspace class, voltage corridor, and weather rank. You only need to rebuild the graph when your underlying data changes.

**Stage 2 — Route & visualise (`app/streamlit_app.py`)**

Loads the serialised graph, re-weights edges in milliseconds via `apply_weights()` based on your slider inputs, runs Yen k-shortest pathfinding, and renders candidate routes on an interactive Folium map. No rebuild required between queries.

**Edge weight function:**
```
weight = λ_t · time_min + λ_e · elev_gain_m + λ_n · noise_score + airspace_penalty
```

Hard exclusions (weight = ∞): Class B/C/D airspace, TFRs, DC SFRA, and weather above the selected minima. A 25 nm exemption zone around origin and destination allows routing into and out of controlled airspace at endpoints.

---

## Repository layout

```
trailblazer/
├── build_graph.py          Stage 1 entry point
├── cli.py                  Headless route CLI (smoke testing)
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── eia/                EIA transmission line shapefiles
│   ├── fcc/                FCC ASR CO.dat + RA.dat (cell towers)
│   ├── nasr/               FAA NASR APT/AWY/FIX + class_airspace/
│   ├── population/         LandScan USA raster (pending — Phase 4)
│   └── tfrs.json           TFR polygons (auto-created on first run)
├── trailblazer/
│   ├── types.py            Core dataclasses
│   ├── elevation.py        SRTM elevation fetcher (Open-Topo-Data)
│   ├── infra/              EIA transmission + FCC tower parsers
│   ├── airspace/           FAA class airspace + TFR exclusion
│   ├── weather/            WeatherProvider protocol + G-AIRMET adapter
│   ├── routing/            Graph construction + Yen pathfinding
│   ├── nasr/               NASR NAV/AWY/FIX parser
│   ├── scoring/            Population noise scoring (stub — Phase 4)
│   └── export/             GeoJSON + Markdown route brief exporter
└── app/
    └── streamlit_app.py    Interactive route planner
```

---

## Getting started

### 1. Install dependencies

```bash
git clone https://github.com/RobbieLBT/trailblazer.git
cd trailblazer
pip install -r requirements.txt
```

Trailblazer also imports from the sibling [Weatherboy](https://github.com/RobbieLBT/weatherboy) repo for METAR interpolation. Clone it alongside Trailblazer and no pip install is required — the path is injected automatically.

### 2. Download data

Place data in the directories shown above. See the **Data sources** section below for download links.

At minimum you need:
- EIA transmission line shapefile → `data/eia/`
- FCC ASR `CO.dat` + `RA.dat` → `data/fcc/`
- FAA NASR class airspace shapefiles → `data/nasr/class_airspace/` *(critical for correct airspace exclusion)*

Optional but recommended:
- G-AIRMET cache: `wget -O data/gairmets.cache.xml.gz https://aviationweather.gov/data/cache/gairmets.cache.xml.gz`

### 3. Build the graph

```bash
# Baseline transmission graph
python build_graph.py --graph transmission

# With airspace exclusion (requires NASR shapefiles)
python build_graph.py --graph transmission --airspace

# Full production build (airspace + SRTM elevation + G-AIRMET weather)
python build_graph.py --graph transmission --airspace --elevation --wx gairmet

# Cell tower mesh
python build_graph.py --graph cell

# NASR Victor airway comparison graph
python build_graph.py --graph nasr --nasr data/nasr
```

### 4. Launch the app

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar to set origin/destination, adjust routing weights (time vs. elevation vs. noise), set weather minima, and inject TFRs. Routes are ranked in a table with ETE, Class E exposure time, total climb, and a one-click export to GeoJSON/Markdown brief.

### 5. Headless smoke test

```bash
python cli.py --graph transmission --origin ORF --dest CRW --wx none
```

---

## Data sources

| Layer | Source |
|-------|--------|
| **EIA Transmission Lines** (HIFLD) | [HIFLD Open Data on ArcGIS Hub](https://hub.arcgis.com/search?tags=hifld) |
| **FCC ASR Cell Towers** (CO.dat + RA.dat) | [FCC Public Access Files](https://www.fcc.gov/wireless/data/public-access-files-database-downloads) — download `l_asr.zip` |
| **LandScan USA Population Raster** (Phase 4) | [ArcGIS — LandScan USA](https://www.arcgis.com/home/item.html?id=d4090758322c4d32a4cd002ffaa0aa12&sublayer=0#visualize) |
| **FAA NASR Subscription** (airspace shapefiles, APT, AWY, FIX) | [FAA NASR Subscription 2026-05-14](https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/2026-05-14/) |

---

## Current status

| Phase | Status |
|-------|--------|
| Transmission corridor graph | ✅ Done |
| Airspace exclusion — polygon intersection | ✅ Done |
| Airport buffers (correct boundaries) | 🔶 Needs NASR shapefiles |
| TFR injection (live sidebar) | ✅ Done |
| Cell tower mesh | ✅ Done |
| Social / noise scoring | 🔷 Stub (needs LandScan) |
| Streamlit map + routing | ✅ Done |
| Weather animation (T+0/+30/+1h) | 🔷 Not started |
| Pareto frontier visualisation | 🔷 Not started |

---

## Weather integration

Trailblazer supports two weather providers, selectable at build/query time:

- **G-AIRMET** — reads `data/gairmets.cache.xml.gz` (refreshed from AWC every 5 min). Overlay is rendered on the map; routing penalties for advisory polygons are planned for the next session.
- **Weatherboy** — full METAR interpolation from Iowa State Mesonet. Activate with `--wx weatherboy --wb-config ../weatherboy/config/virginia.xml`.