# Urban Service Equity — Demo Talking Points

## What the Project Is

A data-driven web dashboard for **San Francisco** that identifies **service equity gaps** at the city-block (grid) level by combining:
- **311 service request data** — 367k cases (potholes, graffiti, encampments, etc.)
- **Housing inventory data** — 253k rental units across 41 neighborhoods

The key insight: some areas have high housing need but slow or poor city service delivery. The dashboard makes these invisible gaps visible.

---

## Core Analytical Pipeline

The Python pipeline (`run_pipeline.py`) processes `merged_rent_311.csv` through these stages:

1. **Aggregation** — Groups housing units by `grid_id` (city block), computes per-grid medians (rent, resolution days, 311 volume, density, etc.)
2. **K-Means Clustering** — Groups ~1,300–2,000 grid cells into 4 archetypes using 6 features (rent, resolution speed, property age, request intensity, service mix)
3. **Feature Engineering** — Builds two composite indices:
   - **Service Performance (S1–S4)**: volume, resolution speed, service diversity, positive vs. negative request mix
   - **Service Need (N1–N5)**: density, crowding, property age, affordability, tenure stability
4. **Equity Scoring (PCA-based)** — Combines performance and need into a single **0–100 equity score** per grid cell (higher = better served relative to need)
5. **LISA Spatial Autocorrelation** — K-nearest-neighbor spatial clustering identifies 4 quadrant types:
   - **LL (red)**: Low-equity surrounded by low-equity → concentrated hotspot
   - **LH (amber)**: Low-equity isolated in a well-served area → struggling pocket
   - **HH (green)**: Well-served cluster
   - **HL (blue)**: Well-served outlier in an underserved zone
6. **Export** — Outputs `grid_points.geojson`, `metadata.json`, `grid_point_advice.json`, and CSVs

---

## The Frontend Dashboard

Static HTML/JS site served from `docs/`, deployed to Vercel.

**Three map views** (`docs/assets/app.js`):
- **Equity Score** — Red-to-green gradient per grid cell
- **LISA Quadrant** — Color-coded spatial clusters (LL/LH/HH/HL)
- **Cluster** — K-Means archetype colors

**Click any point** → side panel shows: grid ID, neighborhood, top 3 distinguishing z-score features, link to report

**Four report panels:**

| Report | What It Shows |
|--------|---------------|
| Equity Score Report | Neighborhoods ranked by count of statistically significant underserved cells (LL + LH) |
| Cluster Report | Z-score bar chart of one cluster vs. city average + "Dire Needs" + intervention queue |
| LISA Quadrant Report | Spatial cluster drill-down |
| Grid Cell Lookup | Type a grid ID → reverse-geocodes to street name via OpenStreetMap |

**AI Chat panel** (`docs/assets/chat.js`) — RAG-powered Q&A grounded in indexed academic papers with inline citations.

---

## The RAG Backend

Deployed as a Vercel serverless function (`api/chat.ts`):

1. User asks a question
2. Question is embedded via OpenAI `text-embedding-3-small`
3. Cosine similarity search against **Supabase pgvector** table of indexed paper chunks
4. Top 8 chunks sent as context to **GPT-4o**
5. Response with `[ref:1]` style citations returned to chat panel

Papers are pre-indexed offline via `rag/index_papers.py`.

---

## Live Demo Flow (Suggested Order)

1. **Hero section** → set the scene: 367k cases, 253k units, 41 neighborhoods
2. **Switch map views** → show equity gradient → LISA quadrants → clusters
3. **Click a red/LL cell** → show side panel with top 3 z-score features explaining *why* it's underserved
4. **Equity Score Report** → show neighborhood rankings; pick a neighborhood → select a LL cell → show intervention queue
5. **Cluster Report** → drill into "Slow Resolution Hotspot" cluster → z-score bar chart shows it's +5.5σ above city mean on resolution days
6. **AI Chat** → ask "What research supports equity interventions in high-density areas?" → show paper citations
7. **Close with the hook**: This is actionable — each grid cell gets specific, priority-ranked operational interventions, not just a score

---

## Key Numbers for the Pitch

| Metric | Value |
|--------|-------|
| 311 cases analyzed | 367,000 |
| Housing units analyzed | 253,000 |
| Neighborhoods covered | 41 |
| Avg service resolution time | 11.8 days |
| Grid cells scored | ~1,300–2,000 |
| K-Means clusters | 4 |
| Spatial analysis | LISA with 999 permutations, p ≤ 0.05 |

---

## Pre-Demo Checklist

- [ ] `merged_rent_311.csv` available locally
- [ ] Run `python run_pipeline.py --input ./merged_rent_311.csv --output-dir docs/outputs --write-geojson`
- [ ] Serve via `python -m http.server 5173 --directory docs` (not `file://`)
- [ ] Confirm `.env.local` has `OPENAI_API_KEY` + `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` if using local RAG chat
