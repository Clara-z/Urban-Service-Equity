# Urban Service Equity Dashboard

## Start Here (Non-Technical User Guide)

If you only want to use the website and do not care about code, read this section only.

### What this website helps you do

- Find areas in San Francisco with lower service equity.
- Compare neighborhoods and cluster patterns.
- See likely root causes and suggested interventions.
- Ask natural-language questions in the built-in chat panel.

### Open the website

- **Public site (GitHub Pages):** open your project URL and go to `index.html`.
- **Local run:** from the project root:

```bash
python -m http.server 5173 --directory docs
```

Then open [http://localhost:5173](http://localhost:5173).

### How to use the dashboard (quick walkthrough)

1. **Start on the map**
   - Use **Color** to switch views (Equity score / LISA / Cluster).
   - Use **Cluster filter** and **Equity min/max** to narrow what you see.
   - Click **Apply** after changing filters.

2. **Click a map point**
   - The right-side panel updates with details for that grid cell.
   - Use **View Equity Score Report** to jump to relevant report sections.

3. **Read reports**
   - **Equity Score Report:** neighborhoods with concentrated low-equity cells.
   - **Needs and Intervention (By Equity Score):** focused guidance for selected low-equity cells.
   - **LISA Quadrant Report:** local spatial pattern interpretation (LL/LH/HH/HL).
   - **Cluster Report:** cluster-level stats, root-cause profile, and context.

4. **Use chat while exploring**
   - Floating chat is on the page; ask plain-language questions.
   - Good prompts:
     - "What is the most urgent issue in this selected area?"
     - "Compare this cluster with city average in simple language."
     - "What interventions should be prioritized first and why?"

### What each view means (plain language)

- **Equity score (0-100):** higher usually means better balance between service outcomes and neighborhood need.
- **LISA quadrant:** whether low/high equity areas are surrounded by similar or different neighbors.
- **Cluster:** groups of areas that behave similarly across multiple indicators.

### Troubleshooting (user-level)

- **Map is blank:** data files are probably missing from `docs/outputs/`.
- **Chat not answering:** backend endpoint or keys may not be configured yet.
- **Numbers look outdated:** pipeline outputs may need to be regenerated.

---

## Operator Quick Start (Run/Refresh Data)

Use this section if you are helping maintain the website data.

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ is recommended.

### 2) Run pipeline and generate website artifacts

```bash
python run_pipeline.py --input /path/to/merged_rent_311.csv --output-dir docs/outputs --write-geojson
```

### 3) Serve locally

```bash
python -m http.server 5173 --directory docs
```

### 4) Validate expected files exist

You should see outputs like:

- `docs/outputs/grid_points.geojson`
- `docs/outputs/cluster_summary.csv`
- `docs/outputs/cluster_feature_zscores.csv`
- `docs/outputs/metadata.json`
- `docs/outputs/grid_point_advice.json`

### 5) Rebuild place mapping when geometry files change

If you update:

- `docs/outputs/grid_points.geojson`
- `docs/outputs/sf_neighborhoods.geojson`
- `docs/outputs/sf_supervisor_districts.geojson`

run:

```bash
python3 scripts/build_grid_place_map.py
```

---

## Project Purpose

This project combines housing and 311 service data to build a block-level equity diagnostic for San Francisco.
The website is static and reads precomputed artifacts; all heavy computation is done in the pipeline.

---

## Repository Structure

| Path | Purpose |
|------|---------|
| `run_pipeline.py` | End-to-end processing and artifact generation |
| `docs/` | Static web app (map, reports, chat UI) |
| `api/chat.ts` | RAG API endpoint (for deployed backend use) |
| `rag/README.md` | RAG indexing and retrieval setup notes |
| `requirements.txt` | Python dependencies |

---

## Data and Artifacts

### Build artifacts (regenerated, not versioned)

Folders like `outputs/`, `outputs_full/`, and `docs/outputs/` are generated artifacts and are ignored by git.

### Typical generated files

- `grid_results.csv`: scored grid-level table
- `cluster_summary.csv`: per-cluster descriptive stats
- `cluster_feature_zscores.csv`: cluster vs city feature differences
- `metadata.json`: PCA weights, top features, heuristics
- `grid_point_advice.json`: point-level needs/actions
- `grid_points.geojson`: map points (when geojson export is enabled)
- `grid_place_map.csv`: lookup for place-aware interactions

### Optional enrichment inputs

When available, these can enrich point-level advice:

- `data/rent_dataset_module2.csv`
- `data/grid_level_rent_311.csv`
- `data/311_data.csv`

Missing optional files are skipped by design.

---

## Input Expectations

`run_pipeline.py` expects a merged unit-level dataset with required housing and 311-related columns (including `grid_id` and location fields).
Large datasets are typically kept out of git and provided through local paths or cloud storage.

---

## Technical Appendix

### Analytics design (high level)

- **Equity score:** composite index from service-performance and service-need components.
- **Clustering:** K-Means groups grid cells into interpretable archetypes.
- **Root-cause layer:** z-score contrasts identify where each cluster diverges from city baseline.
- **Heuristics:** precomputed intervention recommendations attach a policy narrative.

### Frontend architecture

- Static app under `docs/` (no backend required for map/report rendering).
- Browser loads generated CSV/JSON/GeoJSON artifacts.
- Map + report + chat are linked through shared client-side state/events.

### RAG hosting story (Vercel + Supabase)

This project’s chat experience is designed to support paper-grounded answers through a hosted retrieval pipeline:

1. **Document indexing**
   - Research PDFs are chunked and embedded.
   - Chunks + embeddings are stored in **Supabase** with pgvector.

2. **Retrieval API**
   - A server endpoint (deployed on **Vercel**) receives user questions.
   - It performs vector retrieval from Supabase and assembles relevant context.

3. **Generation**
   - The endpoint calls the selected model with retrieved context + dashboard context.
   - It returns answer text plus citations metadata.

4. **Client behavior**
   - The website chat calls the deployed API endpoint (`/api/chat`) rather than calling model providers directly from the browser.

Important note: deployment/settings can be adjusted later (keys, env vars, model choices, retrieval tuning). This README describes the intended hosted architecture without changing current code/config.

For deeper setup details, start with `rag/README.md`.

---

## License / Attribution

Add your course/team attribution and license text here.
