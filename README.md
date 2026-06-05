# Trailblazer

**Intelligent pathfinding for BVLOS UAS operations**

Trailblazer is a two-stage Python framework that builds weighted routing graphs from real infrastructure and airspace data, then finds and visualises optimal BVLOS (Beyond Visual Line of Sight) drone routes through an interactive Streamlit app. It was developed by Commonwealth Drone Solutions.

---

## How it works

**Stage 1 — Graph build (`build_graph.py`)**

Ingests infrastructure, airspace, elevation, weather, and population data and serialises a weighted [NetworkX](https://networkx.org/) graph to a `.pkl` file. Each edge stores attributes including flight time, elevation gain, airspace class, voltage corridor, weather rank, and raw population buffer sum. You only need to rebuild the graph when your underlying data changes.

**Stage 2 — Route & visualise (`app/streamlit_app.py`)**

Loads the serialised graph, re-weights edges in milliseconds via `apply_weights()` based on your slider inputs, runs Yen k-shortest pathfinding, and renders candidate routes on an interactive Folium map with an optional population heatmap overlay. No rebuild required between queries.

**Edge weight function:**
```
weight = λ_t · time_min + λ_e · elev_gain_m + λ_n · noise_score + airspace_penalty
```

`noise_score = log1p(pop_sum / cruise_kts)`, normalised [0,1] across all edges at query time. Changing cruise speed or the λ_n slider re-ranks corridors instantly without a rebuild. A voltage ROW discount is applied to `noise_score` for transmission edges (765 kV → 0.10×, down to off-ROW → 1.00×).

Hard exclusions (weight = ∞): Class B/C/D airspace, TFRs, DC SFRA, and weather above the selected minima. A 10 nm exemption zone around origin and destination allows routing into and out of controlled airspace at endpoints.

---

## Repository layout

```
trailblazer/
├── build_graph.py          Stage 1 entry point
├── cli.py                  Headless route CLI (smoke testing)
├── sweep.py                Batch parameter sweep framework
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── eia/                EIA transmission line shapefiles
│   ├── fcc/                FCC ASR CO.dat + RA.dat (cell towers)
│   ├── nasr/               FAA NASR APT/AWY/FIX + class_airspace/
│   ├── population/         LandScan USA raster (data use agreement — see below)
│   └── tfrs.json           TFR polygons (auto-created on first run)
├── trailblazer/
│   ├── types.py            Core dataclasses
│   ├── elevation.py        SRTM elevation fetcher (Open-Topo-Data)
│   ├── infra/              EIA transmission + FCC tower parsers
│   ├── airspace/           FAA class airspace + TFR exclusion
│   ├── weather/            WeatherProvider protocol + G-AIRMET adapter
│   ├── routing/            Graph construction + Yen pathfinding
│   ├── nasr/               NASR NAV/AWY/FIX parser
│   ├── scoring/            Population impact scoring (LandScan USA, rasterstats)
│   └── export/             GeoJSON, KML (Weatherboy), and Markdown route brief
└── app/
    ├── streamlit_app.py    Interactive route planner
    └── static/tiles/       Pre-rendered network tile PNGs
output/
    └── analyze_sweep.py    Sweep results visualisation
```

---

## Getting started

### 1. Install dependencies

```bash
git clone https://github.com/RobbieLBT/trailblazer.git
cd trailblazer
pip install -r requirements.txt       # includes rasterstats for population scoring
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
- LandScan USA GeoTIFF → `data/population/` *(required for noise scoring and population overlay)*

### 3. Build the graph

```bash
# Baseline transmission graph
python build_graph.py --graph transmission

# With airspace exclusion (requires NASR shapefiles)
python build_graph.py --graph transmission --airspace

# Full production build (airspace + elevation + weather + population scoring)
python build_graph.py --graph transmission --airspace --elevation --wx gairmet \
    --landscan data/population/landscan-mosaic-unitedstates-v1-assets/landscan-mosaic-unitedstates-v1.tif

# Cell tower mesh
python build_graph.py --graph cell

# NASR Victor airway comparison graph
python build_graph.py --graph nasr --nasr data/nasr
```

`--landscan` triggers population buffer sampling (rasterstats, ~60 s) and generates a `population_<name>.png` heatmap overlay alongside the pkl. Graphs built without `--landscan` run normally; the noise slider is disabled in the app.

### 4. Launch the app

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar to set origin/destination, adjust routing weights (λ_t time, λ_e elevation, λ_n noise/population), set weather minima, and inject TFRs. Routes are ranked in a table with ETE, people within 250 m buffer, Class E exposure time, total climb, and a one-click export to GeoJSON/Markdown brief.

### 5. Headless smoke test

```bash
python cli.py --graph transmission --origin ORF --dest CRW --wx none
```

### 6. Batch sweep

Compare graph types and weight sensitivity; results written to CSV:

```bash
python sweep.py \
    --pkls graph_cell.pkl graph_transmission.pkl \
    --origin ORF --dest CRW \
    --noise-weights 0.0 0.1 0.2 0.3 0.5 0.75 1.0 \
    --cruise-kts 80 100 120 140 \
    --output output/sensitivity.csv

python output/analyze_sweep.py output/sensitivity.csv
```

### 7. Weatherboy export

Export a route as KML for Weatherboy mission traversal:

```python
from trailblazer.export.export import write_kml
write_kml(route_set, "output/ORF_CRW_cell_rank1.kml")   # rank-1 default
```

```bash
# In Weatherboy
python3 run.py --config config/virginia.xml \
               --path output/ORF_CRW_cell_rank1.kml \
               --alt-agl 350 --speed-kmh 65 \
               --output output/ORF_CRW_cell_rank1_forcing.csv \
               --no-animate
```

The resulting `ForcingRecord` CSV (wind, gust, headwind, sideslip, density, flight category at 10 s intervals) feeds directly into Phase 5 vehicle dynamics integration.

---

## Data sources

| Layer | Source |
|-------|--------|
| **EIA Transmission Lines** (HIFLD) | [HIFLD Open Data on ArcGIS Hub](https://hub.arcgis.com/search?tags=hifld) |
| **FCC ASR Cell Towers** (CO.dat + RA.dat) | [FCC Public Access Files](https://www.fcc.gov/wireless/data/public-access-files-database-downloads) — download `l_asr.zip` |
| **LandScan USA Population Raster** | [ArcGIS — LandScan USA](https://www.arcgis.com/home/item.html?id=d4090758322c4d32a4cd002ffaa0aa12) — data use agreement required; not redistributable |
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
| Population / noise scoring (LandScan USA) | ✅ Done |
| Population heatmap overlay (Streamlit) | ✅ Done |
| KML export → Weatherboy traversal handoff | ✅ Done |
| Batch sweep framework + visualisation | ✅ Done |
| Streamlit map + routing | ✅ Done |
| Tango — air traffic density surface | 🔷 Not started |
| Weather animation (T+0/+30/+1h) | 🔷 Not started |
| Pareto frontier visualisation | 🔷 Not started |

---

## Weather integration

Trailblazer supports two weather providers, selectable at build/query time:

- **G-AIRMET** — reads `data/gairmets.cache.xml.gz` (refreshed from AWC every 5 min). Overlay is rendered on the map; IFR and MTN OBSCN advisories above the configurable threshold become hard routing exclusions.
- **Weatherboy** — full METAR interpolation from Iowa State Mesonet. Activate with `--wx weatherboy --wb-config ../weatherboy/config/virginia.xml`