import { ragChat } from "./rag_client.js";

const DATA_BASE = new URL("../outputs/", import.meta.url);
const DATA_GEOJSON = new URL("grid_points.geojson", DATA_BASE).href;
const DATA_META = new URL("metadata.json", DATA_BASE).href;
const DATA_SUMMARY = new URL("cluster_summary.csv", DATA_BASE).href;
// Main detailed rent/311 context source for chat.
const DATA_RENT_CSV = new URL("rent_listings_slim.csv", DATA_BASE).href;
const DATA_RENT_FALLBACK_CSV = new URL("merged_rent_311.csv", DATA_BASE).href;
const DATA_RENT_LEGACY_CSV = new URL("rent_dataset_module2.csv", DATA_BASE).href;
const DATA_GRID_RENT_311_CSV = new URL("grid_level_rent_311.csv", DATA_BASE).href;
const SOCIOPAPER_TXT = new URL("../sociopaper/zahnow1.txt", import.meta.url).href;

const STORAGE_KEY = "equity_prompt_lab_v1";
const MAX_HISTORY = 16;
const SOCIOPAPER_ID = "zahnow1";
const SOCIOPAPER_TOP_K = 5;
const SOCIOPAPER_CHUNK_TARGET = 1100;
const RENT_CONTEXT_MAX_ROWS = 22;
const RENT_LOAD_MAX_ROWS = 80000;
const GRID_LOOKUP_SAMPLE_ROWS = 2;
const QUIET_FRIENDLY_HOODS = new Set([
  "Outer Richmond",
  "Inner Richmond",
  "Sunset/Parkside",
  "Lakeshore",
  "Seacliff",
  "Oceanview/Merced/Ingleside",
  "West of Twin Peaks",
  "Visitacion Valley",
  "Bayview Hunters Point",
]);
const BUSIER_HOODS = new Set([
  "Tenderloin",
  "South of Market",
  "Financial District/South Beach",
  "Mission",
  "Chinatown",
  "North Beach",
]);
/** Canonical analysis_neighborhood values in merged_rent_311 (keep in sync with data). */
const SF_RENT_NEIGHBORHOODS = [
  "Bayview Hunters Point",
  "Bernal Heights",
  "Castro/Upper Market",
  "Chinatown",
  "Excelsior",
  "Financial District/South Beach",
  "Glen Park",
  "Golden Gate Park",
  "Haight Ashbury",
  "Hayes Valley",
  "Inner Richmond",
  "Inner Sunset",
  "Japantown",
  "Lakeshore",
  "Lincoln Park",
  "Lone Mountain/USF",
  "Marina",
  "McLaren Park",
  "Mission",
  "Mission Bay",
  "Nob Hill",
  "Noe Valley",
  "North Beach",
  "Oceanview/Merced/Ingleside",
  "Outer Mission",
  "Outer Richmond",
  "Pacific Heights",
  "Portola",
  "Potrero Hill",
  "Presidio",
  "Presidio Heights",
  "Russian Hill",
  "Seacliff",
  "South of Market",
  "Sunset/Parkside",
  "Tenderloin",
  "Treasure Island",
  "Twin Peaks",
  "Visitacion Valley",
  "West of Twin Peaks",
  "Western Addition",
];
const STOPWORDS = new Set([
  "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "has", "have", "been", "were", "said", "each",
  "which", "their", "time", "will", "about", "there", "could", "other", "than", "then", "them", "these", "some", "what", "with", "from", "that",
  "this", "into", "such", "when", "may", "more", "also", "how", "its", "who", "had", "any",
]);

const els = {
  floatingChatDock: document.getElementById("floatingPublicChat"),
  floatingChatToggle: document.getElementById("publicChatToggle"),
  reportCluster: document.getElementById("reportCluster"),
  selGridId: document.getElementById("selGridId"),
  modelSelect: document.getElementById("modelSelect"),
  customModel: document.getElementById("customModel"),
  apiKey: document.getElementById("apiKey"),
  ragApiBase: document.getElementById("ragApiBase"),
  temperature: document.getElementById("temperature"),
  maxTokens: document.getElementById("maxTokens"),
  topK: document.getElementById("topK"),
  systemPrompt: document.getElementById("systemPrompt"),
  contextPreview: document.getElementById("contextPreview"),
  chatMessages: document.getElementById("chatMessages"),
  userInput: document.getElementById("userInput"),
  sendBtn: document.getElementById("sendBtn"),
  clearChat: document.getElementById("clearChat"),
  chatStatus: document.getElementById("chatStatus"),
};

const state = {
  chat: [],
  meta: null,
  geo: null,
  summaryRows: [],
  gridById: new Map(),
  summaryByCluster: new Map(),
  /** @type {{ id: number; text: string }[]} */
  sociopaperChunks: [],
  /** @type {{ addr: string; hood: string; grid_id: string; rent: number; beds: number | null; baths: number | null; sqft: number | null; year: number | null; district: number | null; total_311_requests: number | null; top_service: string | null; pct_parking_enforcement: number | null; pct_mta_parking_traffic_signs_high_priority: number | null; pct_mta_parking_traffic_signs_normal_priority: number | null }[]} */
  rentListings: [],
  /** @type {Map<string, any>} */
  mergedGridIndex: new Map(),
  /** @type {{ grid_id: string; neighborhood: string | null; summary: Record<string, any> }[]} */
  gridRent311Rows: [],
  /** @type {Map<string, { neighborhood: string | null; summary: Record<string, any> }>} */
  gridRentById: new Map(),
  /** @type {Map<string, { n_cases: number; top_type: string | null; top_share: number | null }>} */
  caseSummaryByGrid: new Map(),
  /** @type {Map<string, { n_cases: number; top_type: string | null; top_share: number | null }>} */
  caseSummaryByCluster: new Map(),
};

// const DEFAULT_SYSTEM_PROMPT = `You are the assistant inside an urban service equity dashboard, but you behave like a normal helpful chat model: answer the user's actual question in a direct, natural tone.

// Refusals (forbidden):
// - Do NOT answer with stiff scope refusals such as "I can only help with urban service equity" or "ask me about urban services" when the user asks something else. Never deflect harmless test questions.

// How to answer:
// - Questions about this map, clusters, grids, fairness, services, or San Francisco housing in context: use the JSON context below; be concrete; say when something is uncertain.
// - General questions (definitions, machine learning, statistics, what a paper says, etc.): answer straight. Use normal technical vocabulary when it helps. The server also attaches retrieved "Paper excerpts" with [ref:n] labels—use them when relevant and cite like [ref:1]; if excerpts are off-topic, say that in one short clause and answer from general knowledge.
// - Optional sociology text (zahnow1 chunks) in the system message: use for social/urban questions when relevant; cite [zahnow1 chunk N]. Do not force paper citations for unrelated questions.

// Style:
// - Sound like a knowledgeable colleague, not a policy notice.
// - Keep answers concise. Avoid crutch phrases ("I'm here to assist…").
// - Do not use bold-quoted emphasis like **"..."**.`;

const DEFAULT_SYSTEM_PROMPT = `You are an urban service equity assistant.
Your job is to answer user questions clearly using dashboard data and policy reasoning.

Rules:
1) Separate your answer into: place-specific analysis, cluster-level analysis, and general recommendations.
2) If evidence is uncertain, say what is uncertain.
3) Cite concrete references when provided in context.
4) Keep explanations concise and actionable.
5) When sociology paper excerpts are included in the system message, ground relevant conceptual claims in those passages and cite them (e.g. zahnow1 chunk 3).

You are an assistant with access to a sociology paper (zahnow1).

Before answering, you must decide:

1. Does the user's question require sociological reasoning?
   - Yes if it involves:
     - social behavior, communities, inequality, urban dynamics
     - concepts like collective efficacy, social interaction, etc.
   - No if it is:
     - purely technical (coding, math, API usage)
     - factual lookup without social interpretation

2. If YES:
   - Use the provided zahnow1 chunks when relevant
   - Integrate concepts or findings from the paper
   - Cite as [zahnow1 chunk N] when used

3. If NO:
   - Ignore zahnow1 completely
   - Answer normally

4. Never force citations if irrelevant`;


const RESPONSE_STYLE_GUARD = `Formatting:
- No bold-quoted emphasis patterns.

When (and only when) explaining THIS dashboard's JSON, map, cluster report, or SF housing-inventory rows for a public audience:
- Paraphrase internal metric names instead of reciting field names; avoid dumping raw JSON keys.
- Avoid naming schema-style cluster labels when plain language will do.
- If SF housing inventory rows are in context, use only those rows for address-level examples and note they are inventory records, not a live rental feed.

For questions that are not about this dashboard, ignore the paragraph above if it would block a clear answer.`;

function setStatus(msg) {
  if (!els.chatStatus) return;
  els.chatStatus.textContent = msg;
}

function loadSavedState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    els.modelSelect.value = saved.model ?? els.modelSelect.value;
    els.customModel.value = saved.customModel ?? "";
    els.apiKey.value = saved.apiKey ?? "";
    els.ragApiBase.value = saved.ragApiBase ?? (window.RAG_API_BASE || "https://urban-service-equity.vercel.app");
    els.temperature.value = String(saved.temperature ?? 0.3);
    els.maxTokens.value = String(saved.maxTokens ?? 900);
    els.topK.value = String(saved.topK ?? 8);
    els.systemPrompt.value = saved.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
    state.chat = Array.isArray(saved.chat)
      ? saved.chat.slice(-MAX_HISTORY).map((m) => ({
          ...m,
          content: m?.role === "assistant" ? sanitizeAssistantText(m?.content) : String(m?.content ?? ""),
        }))
      : [];
  } catch {
    // ignore malformed local storage
  }
}

function saveState() {
  const payload = {
    model: els.modelSelect.value,
    customModel: els.customModel.value,
    apiKey: els.apiKey.value,
    ragApiBase: els.ragApiBase.value,
    temperature: Number(els.temperature.value),
    maxTokens: Number(els.maxTokens.value),
    topK: Number(els.topK.value),
    systemPrompt: els.systemPrompt.value,
    chat: state.chat.slice(-MAX_HISTORY),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function selectedGridId() {
  const raw = String(els.selGridId?.textContent ?? "").trim();
  if (!raw || raw === "—" || raw.toLowerCase() === "none") return "";
  return normalizeGridId(raw);
}

function reportClusterId() {
  const v = String(els.reportCluster?.value ?? "").trim();
  return v || "";
}

function isGridSelected() {
  const gid = selectedGridId();
  return Boolean(gid) && state.gridById.has(gid);
}

function currentContextSelection() {
  if (isGridSelected()) {
    const gid = selectedGridId();
    const row = state.gridById.get(gid) ?? null;
    return { mode: "grid", gridId: gid, clusterId: row?.cluster != null ? String(row.cluster) : reportClusterId() || "0" };
  }
  const clusterId = reportClusterId();
  if (clusterId) return { mode: "cluster", gridId: "", clusterId };
  return { mode: "global", gridId: "", clusterId: "0" };
}

function selectedModel() {
  const custom = (els.customModel.value || "").trim();
  return custom || els.modelSelect.value;
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} loading ${url}`);
  return r.json();
}

function parseCsv(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err),
    });
  });
}

function normalizeGridId(v) {
  const s = String(v ?? "").trim();
  if (s.endsWith(".0") && /^\d+\.0$/.test(s)) return s.slice(0, -2);
  return s;
}

function pickFirstColumnName(row, names) {
  if (!row || typeof row !== "object") return null;
  const keys = Object.keys(row);
  const lowerMap = new Map(keys.map((k) => [k.toLowerCase(), k]));
  for (const n of names) {
    const hit = lowerMap.get(String(n).toLowerCase());
    if (hit) return hit;
  }
  return null;
}

function numericOrNull(v) {
  const n = typeof v === "number" ? v : Number(String(v ?? "").replace(/,/g, "").trim());
  return Number.isFinite(n) ? n : null;
}

function summarizeNumeric(values) {
  const xs = values.filter((v) => Number.isFinite(v));
  if (!xs.length) return null;
  const sorted = [...xs].sort((a, b) => a - b);
  const count = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / count;
  const median = count % 2 === 1 ? sorted[(count - 1) / 2] : (sorted[count / 2 - 1] + sorted[count / 2]) / 2;
  return {
    count,
    mean: Math.round(mean * 100) / 100,
    median: Math.round(median * 100) / 100,
    min: sorted[0],
    max: sorted[count - 1],
  };
}

function compactMergedRow(row) {
  const out = {};
  for (const [k, v] of Object.entries(row || {})) {
    if (v === null || v === undefined || v === "") continue;
    if (typeof v === "number" && !Number.isFinite(v)) continue;
    out[k] = v;
  }
  return out;
}

function loadRentDataset() {
  return loadRentDatasetFromUrl(DATA_RENT_CSV);
}

function loadRentDatasetFromUrl(url) {
  return new Promise((resolve, reject) => {
    const rows = [];
    const byGrid = new Map();
    Papa.parse(url, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      step: (res) => {
        const r = res.data;
        if (!r || typeof r !== "object") return;
        const gridId = normalizeGridId(r.grid_id);
        const hood = String(r.analysis_neighborhood ?? "").trim();
        const rent = numericOrNull(r.monthly_rent_clean);
        const total311 = numericOrNull(r.total_311_requests);
        const topService = r.top_service ? String(r.top_service) : null;

        if (gridId) {
          if (!byGrid.has(gridId)) {
            byGrid.set(gridId, {
              row_count: 0,
              neighborhoods: new Map(),
              top_services: new Map(),
              rent_values: [],
              total_311_values: [],
              sample_rows: [],
            });
          }
          const acc = byGrid.get(gridId);
          acc.row_count += 1;
          if (hood) acc.neighborhoods.set(hood, (acc.neighborhoods.get(hood) || 0) + 1);
          if (Number.isFinite(rent) && rent > 0) acc.rent_values.push(rent);
          if (Number.isFinite(total311) && total311 >= 0) acc.total_311_values.push(total311);
          if (topService) acc.top_services.set(topService, (acc.top_services.get(topService) || 0) + 1);
          if (acc.sample_rows.length < GRID_LOOKUP_SAMPLE_ROWS) acc.sample_rows.push(compactMergedRow(r));
        }

        const addr = String(r.block_address ?? "").trim();
        if (rows.length < RENT_LOAD_MAX_ROWS && Number.isFinite(rent) && rent > 0 && addr && hood) {
          rows.push({
            addr,
            hood,
            grid_id: gridId,
            rent,
            beds: typeof r.bedrooms_clean === "number" && !Number.isNaN(r.bedrooms_clean) ? r.bedrooms_clean : null,
            baths: typeof r.bathrooms_clean === "number" && !Number.isNaN(r.bathrooms_clean) ? r.bathrooms_clean : null,
            sqft: typeof r.sqft_avg === "number" && !Number.isNaN(r.sqft_avg) ? r.sqft_avg : null,
            year: typeof r.submission_year === "number" && !Number.isNaN(r.submission_year) ? r.submission_year : null,
            district: typeof r.supervisor_district === "number" && !Number.isNaN(r.supervisor_district) ? r.supervisor_district : null,
            total_311_requests: total311,
            top_service: topService,
            pct_parking_enforcement: numericOrNull(r.pct_parking_enforcement),
            pct_mta_parking_traffic_signs_high_priority: numericOrNull(r.pct_mta_parking_traffic_signs_high_priority),
            pct_mta_parking_traffic_signs_normal_priority: numericOrNull(r.pct_mta_parking_traffic_signs_normal_priority),
          });
        }
      },
      complete: () => {
        const gridIndex = new Map();
        for (const [gid, acc] of byGrid.entries()) {
          const neighborhoods = [...acc.neighborhoods.entries()].sort((a, b) => b[1] - a[1]).map(([k, n]) => ({ name: k, n }));
          const topServices = [...acc.top_services.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([k, n]) => ({ name: k, n }));
          gridIndex.set(gid, {
            grid_id: gid,
            row_count: acc.row_count,
            neighborhoods,
            rent_stats: summarizeNumeric(acc.rent_values),
            total_311_requests_stats: summarizeNumeric(acc.total_311_values),
            top_services: topServices,
            sample_rows: acc.sample_rows,
          });
        }
        resolve({ rentRows: rows, gridIndex });
      },
      error: (err) => reject(err),
    });
  });
}

async function loadRentDatasetWithFallback() {
  const candidates = [DATA_RENT_CSV, DATA_RENT_FALLBACK_CSV, DATA_RENT_LEGACY_CSV];
  let lastErr = null;
  for (const url of candidates) {
    try {
      const parsed = await loadRentDatasetFromUrl(url);
      const rows = parsed?.rentRows?.length ?? 0;
      const grids = parsed?.gridIndex?.size ?? 0;
      if (rows > 0 || grids > 0) {
        return { ...parsed, sourceUrl: url };
      }
      lastErr = new Error(`No usable rows in ${url}`);
    } catch (err) {
      lastErr = err;
      console.warn(`rent dataset load failed from ${url}`, err);
    }
  }
  throw lastErr || new Error("No rent dataset source could be loaded.");
}

async function loadGridRent311Dataset() {
  const rows = await parseCsv(DATA_GRID_RENT_311_CSV);
  if (!Array.isArray(rows) || !rows.length) return [];
  const gcol = pickFirstColumnName(rows[0], ["grid_id", "Grid_ID", "GRID_ID", "grid id"]);
  if (!gcol) return [];
  const ncol = pickFirstColumnName(rows[0], ["analysis_neighborhood", "neighborhood", "NEIGHBORHOOD"]);

  const out = [];
  const seen = new Set();
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const gid = normalizeGridId(row[gcol]);
    if (!gid) continue;
    if (seen.has(gid)) continue;
    seen.add(gid);
    const summary = {};
    for (const [k, v] of Object.entries(row)) {
      if (k === gcol || k === ncol) continue;
      if (v === null || v === undefined || v === "") continue;
      if (typeof v === "number" && !Number.isFinite(v)) continue;
      summary[k] = v;
    }
    out.push({
      grid_id: gid,
      neighborhood: ncol ? String(row[ncol] ?? "").trim() || null : null,
      summary,
    });
  }
  return out;
}

function reindexSupplementalData() {
  state.gridRentById.clear();
  state.caseSummaryByGrid.clear();
  state.caseSummaryByCluster.clear();

  for (const row of state.gridRent311Rows ?? []) {
    state.gridRentById.set(String(row.grid_id), { neighborhood: row.neighborhood, summary: row.summary });
  }

  // Build 311 summaries from merged context fields already in grid_points.geojson.
  for (const [gid, p] of state.gridById.entries()) {
    const nCases = Number(p?.total_311_requests);
    const topType = p?.top_service ? String(p.top_service) : null;
    const rec = {
      n_cases: Number.isFinite(nCases) ? Math.max(0, Math.round(nCases)) : 0,
      top_type: topType,
      top_share: null,
    };
    state.caseSummaryByGrid.set(gid, rec);
    const c = p && p.cluster != null ? String(p.cluster) : null;
    if (!c) continue;
    if (!state.caseSummaryByCluster.has(c)) state.caseSummaryByCluster.set(c, { n_cases: 0, typeCounts: new Map() });
    const acc = state.caseSummaryByCluster.get(c);
    acc.n_cases += rec.n_cases;
    if (rec.top_type) acc.typeCounts.set(rec.top_type, (acc.typeCounts.get(rec.top_type) || 0) + rec.n_cases);
  }

  // Overlay/augment from grid_level_rent_311 if it includes extra 311 fields.
  for (const [gid, row] of state.gridRentById.entries()) {
    const s = row?.summary || {};
    const n = Number(s.n_311_cases ?? s.total_311_requests ?? s.case_count ?? NaN);
    const t = s.top_311_type ?? s.top_service ?? s.service_name ?? null;
    if (!state.caseSummaryByGrid.has(gid)) {
      state.caseSummaryByGrid.set(gid, {
        n_cases: Number.isFinite(n) ? Math.max(0, Math.round(n)) : 0,
        top_type: t ? String(t) : null,
        top_share: null,
      });
    } else {
      const rec = state.caseSummaryByGrid.get(gid);
      if (Number.isFinite(n) && (!rec.n_cases || rec.n_cases === 0)) rec.n_cases = Math.max(0, Math.round(n));
      if (!rec.top_type && t) rec.top_type = String(t);
      state.caseSummaryByGrid.set(gid, rec);
    }
  }

  // finalize cluster top types
  for (const [c, acc] of [...state.caseSummaryByCluster.entries()]) {
    let top_type = null;
    let top_count = 0;
    for (const [cat, n] of acc.typeCounts.entries()) {
      if (n > top_count) {
        top_count = n;
        top_type = cat;
      }
    }
    state.caseSummaryByCluster.set(c, {
      n_cases: acc.n_cases,
      top_type,
      top_share: top_count > 0 ? top_count / Math.max(acc.n_cases, 1) : null,
    });
  }
}

function rentIntentFromQuery(q) {
  const s = String(q || "").toLowerCase();
  if (!s.trim()) return false;
  const wantsStats =
    /\b(avg|average|mean|median|typical|price level|price|cost|how much)\b/.test(s) &&
    /\b(rent|rental|apartment|housing|house|home|unit|neighborhood|area)\b/.test(s);
  const hasMoney =
    /\$\s*\d/.test(s) ||
    /\d{3,5}\s*[-–—to]+\s*\d{3,5}/.test(s) ||
    /\b(?:under|below|over|above|less than|more than|at least)\s*\$?\s*\d{3,5}\b/.test(s);
  const housingWords =
    /\b(rent|rental|apartment|housing|lease|landlord|tenant|bedroom|bedrooms|studio|sqft|sq\.?\s*ft|afford|move|live|neighborhood|quiet|location|place|options|listing|unit|flat)\b/.test(
      s,
    );
  const mentionsKnownHood = SF_RENT_NEIGHBORHOODS.some((h) => s.includes(h.toLowerCase())) || /\brichmond\b|\bsunset\b|\bmission\b|\bsoma\b/.test(s);
  if (/\bsf\b|\bsan francisco\b|\bay area\b/.test(s) && housingWords) return true;
  if (wantsStats && (housingWords || mentionsKnownHood)) return true;
  if (mentionsKnownHood && /\b(rent|rental|housing|apartment|house|home|price|cost|mean|average|median)\b/.test(s)) return true;
  if (housingWords && hasMoney) return true;
  if (hasMoney && /\b(live|place|quiet|budget|ideal|looking|want)\b/.test(s)) return true;
  if (/\b(rent|rental|apartment|housing|lease|bedroom|studio)\b/.test(s) && hasMoney) return true;
  if (/\$\s*\d{3,5}\b/.test(s) && /\b(rent|month|budget|afford)\b/.test(s)) return true;
  return false;
}

function incidentIntentFromQuery(q) {
  const s = String(q || "").toLowerCase();
  if (!s.trim()) return false;
  const has311 = /\b311\b|\bservice request\b|\bcomplaint\b|\bincident\b|\bcases?\b/.test(s);
  const hasCategory = /\bparking\b|\bgraffiti\b|\bnoise\b|\bencampment\b|\bstreet\b|\bsidewalk\b/.test(s);
  const mentionsKnownHood = SF_RENT_NEIGHBORHOODS.some((h) => s.includes(h.toLowerCase())) || /\brichmond\b|\bsunset\b|\bmission\b|\bsoma\b/.test(s);
  return (has311 || hasCategory) && (mentionsKnownHood || /\bsf\b|\bsan francisco\b/.test(s));
}

function parseRentRangeFromQuery(q) {
  const s = String(q || "");
  let min = null;
  let max = null;
  const dollarPair = s.match(/\$\s*(\d{3,5})\s*[-–—to,]+\s*\$?\s*(\d{3,5})/i);
  const plainPair = s.match(/\b(\d{3,5})\s*[-–—to]+\s*(\d{3,5})\b/);
  if (dollarPair) {
    min = Number(dollarPair[1]);
    max = Number(dollarPair[2]);
  } else if (plainPair) {
    min = Number(plainPair[1]);
    max = Number(plainPair[2]);
  }
  if (min != null && max != null && min > max) [min, max] = [max, min];
  const under = s.match(/\b(?:under|below|less than|max|maximum)\s*\$?\s*(\d{3,5})\b/i);
  if (under) max = Number(under[1]);
  const over = s.match(/\b(?:over|above|at least|min|minimum)\s*\$?\s*(\d{3,5})\b/i);
  if (over) min = Number(over[1]);
  return { min, max };
}

function parseBedroomsHint(q) {
  const s = String(q || "").toLowerCase();
  if (/\bstudio\b/.test(s)) return 0;
  const m = s.match(/\b(\d)\s*(?:br|bed|bedroom|bedrooms)\b/);
  if (m) return Number(m[1]);
  return null;
}

function neighborhoodsFromQuery(q) {
  const s = String(q || "");
  const lower = s.toLowerCase();
  const found = new Set();
  for (const hood of SF_RENT_NEIGHBORHOODS) {
    if (lower.includes(hood.toLowerCase())) found.add(hood);
  }
  if (/\bsoma\b/.test(lower) || /south of market/i.test(s)) found.add("South of Market");
  if (/\bfidi\b|financial district/i.test(lower)) found.add("Financial District/South Beach");
  if (/\bouter\s+richmond\b|\brichmond\b/.test(lower) && !/\binner\s+richmond\b/.test(lower)) {
    found.add("Outer Richmond");
    found.add("Inner Richmond");
  }
  if (/\binner\s+richmond\b/.test(lower)) found.add("Inner Richmond");
  if (/\bsunset\b|\bparkside\b/.test(lower)) found.add("Sunset/Parkside");
  if (/\bhaight\b/.test(lower)) found.add("Haight Ashbury");
  if (/\bcastro\b|\bupper market\b/.test(lower)) found.add("Castro/Upper Market");
  if (/\bnoe\b/.test(lower)) found.add("Noe Valley");
  if (/\bpotrero\b/.test(lower)) found.add("Potrero Hill");
  if (/\bmission\b/.test(lower) && !/mission bay/i.test(s)) found.add("Mission");
  if (/mission bay/i.test(s)) found.add("Mission Bay");
  return [...found];
}

function quietPreferenceFromQuery(q) {
  return /\bquiet\b|\bcalm\b|\bpeaceful\b|\blow noise\b/i.test(String(q || ""));
}

function retrieveRentListings(userQuery) {
  if (!state.rentListings.length) return "";
  const { min, max } = parseRentRangeFromQuery(userQuery);
  const hoods = neighborhoodsFromQuery(userQuery);
  const bedHint = parseBedroomsHint(userQuery);
  const wantQuiet = quietPreferenceFromQuery(userQuery);

  let candidates = state.rentListings;
  if (hoods.length) {
    const set = new Set(hoods);
    candidates = candidates.filter((r) => set.has(r.hood));
  }
  if (min != null) candidates = candidates.filter((r) => r.rent >= min);
  if (max != null) candidates = candidates.filter((r) => r.rent <= max);
  if (bedHint != null) {
    candidates = candidates.filter((r) => r.beds == null || Math.abs(Number(r.beds) - bedHint) < 0.51);
  }

  if (!candidates.length) {
    candidates = state.rentListings;
    if (min != null) candidates = candidates.filter((r) => r.rent >= min);
    if (max != null) candidates = candidates.filter((r) => r.rent <= max);
    if (bedHint != null) candidates = candidates.filter((r) => r.beds == null || Math.abs(Number(r.beds) - bedHint) < 0.51);
  }

  if (!candidates.length) {
    return `San Francisco housing inventory: no rows matched the parsed filters (budget or bedroom). Ask the user to widen the rent range or remove bedroom filters. Dataset has ${state.rentListings.length} units with rent.`;
  }

  const rentsAll = candidates.map((r) => Number(r.rent)).filter((v) => Number.isFinite(v) && v > 0);
  const count = rentsAll.length;
  const mean = count ? rentsAll.reduce((a, b) => a + b, 0) / count : null;
  const sorted = [...rentsAll].sort((a, b) => a - b);
  const median = !count
    ? null
    : count % 2 === 1
      ? sorted[(count - 1) / 2]
      : (sorted[count / 2 - 1] + sorted[count / 2]) / 2;

  const scored = candidates.map((r) => {
    let score = 0;
    if (wantQuiet) {
      if (QUIET_FRIENDLY_HOODS.has(r.hood)) score += 6;
      if (BUSIER_HOODS.has(r.hood)) score -= 4;
    }
    score -= r.rent / 5000;
    return { r, score };
  });
  scored.sort((a, b) => b.score - a.score || a.r.rent - b.r.rent);

  const picked = [];
  const seen = new Set();
  for (const { r } of scored) {
    const key = `${r.addr}|${r.hood}|${r.rent}`;
    if (seen.has(key)) continue;
    seen.add(key);
    picked.push(r);
    if (picked.length >= RENT_CONTEXT_MAX_ROWS) break;
  }

  const payload = {
    source: "San Francisco merged housing + 311 dataset (merged_rent_311.csv)",
    note: "Rows are city-reported units; not guaranteed to be vacant or on the market today.",
    filters_applied: { min, max, neighborhoods: hoods, bedroom_hint: bedHint, quiet_bias: wantQuiet },
    aggregate_stats: {
      row_count: count,
      mean_monthly_rent_usd: mean == null ? null : Math.round(mean * 100) / 100,
      median_monthly_rent_usd: median == null ? null : Math.round(median * 100) / 100,
      min_monthly_rent_usd: count ? sorted[0] : null,
      max_monthly_rent_usd: count ? sorted[sorted.length - 1] : null,
    },
    listings: picked.map((r, i) => ({
      n: i + 1,
      block: r.addr,
      neighborhood: r.hood,
      monthly_rent_usd: Math.round(r.rent * 100) / 100,
      bedrooms: r.beds,
      bathrooms: r.baths,
      approx_sqft: r.sqft,
      inventory_year: r.year,
      supervisor_district: r.district,
    })),
  };
  return `San Francisco housing inventory sample (use for concrete SF examples only):\n${JSON.stringify(payload, null, 2)}`;
}

function retrieve311Incidents(userQuery) {
  if (!state.rentListings.length) return "";
  const hoods = neighborhoodsFromQuery(userQuery);
  let rows = state.rentListings;
  if (hoods.length) {
    const set = new Set(hoods);
    rows = rows.filter((r) => set.has(r.hood));
  }
  if (!rows.length) return "";

  const byGrid = new Map();
  for (const r of rows) {
    const gid = String(r.grid_id || "");
    if (!gid) continue;
    if (!byGrid.has(gid)) {
      byGrid.set(gid, {
        neighborhood: r.hood,
        total_311_requests: Number.isFinite(r.total_311_requests) ? Number(r.total_311_requests) : 0,
        top_service: r.top_service || null,
        pct_parking_enforcement: Number.isFinite(r.pct_parking_enforcement) ? Number(r.pct_parking_enforcement) : 0,
        pct_mta_parking_traffic_signs_high_priority: Number.isFinite(r.pct_mta_parking_traffic_signs_high_priority)
          ? Number(r.pct_mta_parking_traffic_signs_high_priority)
          : 0,
        pct_mta_parking_traffic_signs_normal_priority: Number.isFinite(r.pct_mta_parking_traffic_signs_normal_priority)
          ? Number(r.pct_mta_parking_traffic_signs_normal_priority)
          : 0,
      });
    }
  }

  let total311 = 0;
  let estimatedParking = 0;
  for (const g of byGrid.values()) {
    const total = Number(g.total_311_requests || 0);
    const parkingShare =
      Number(g.pct_parking_enforcement || 0) +
      Number(g.pct_mta_parking_traffic_signs_high_priority || 0) +
      Number(g.pct_mta_parking_traffic_signs_normal_priority || 0);
    total311 += total;
    estimatedParking += total * Math.max(0, parkingShare);
  }

  const out = {
    source: "San Francisco merged housing + 311 dataset (merged_rent_311.csv)",
    filter_neighborhoods: hoods,
    grid_count: byGrid.size,
    total_311_requests_estimate: Math.round(total311),
    parking_311_incidents_estimate: Math.round(estimatedParking),
    method:
      "Deduplicated by grid_id, then summed total_311_requests * (pct_parking_enforcement + pct_mta_parking_traffic_signs_high_priority + pct_mta_parking_traffic_signs_normal_priority).",
  };
  return `311 incident estimate from merged dataset:\n${JSON.stringify(out, null, 2)}`;
}

function extractGridIdsFromQuery(q) {
  const hits = String(q || "").match(/\b\d+_\d+\b/g);
  if (!hits) return [];
  return [...new Set(hits.map((h) => normalizeGridId(h)))];
}

function retrieveGridCellDetails(userQuery) {
  const gids = extractGridIdsFromQuery(userQuery);
  if (!gids.length || !state.mergedGridIndex.size) return "";
  const found = [];
  const missing = [];
  for (const gid of gids) {
    const rec = state.mergedGridIndex.get(gid);
    if (rec) found.push(rec);
    else missing.push(gid);
  }
  if (!found.length) {
    return `Grid lookup from merged dataset: none of the requested grid_id values were found. Missing: ${missing.join(", ")}`;
  }
  const payload = {
    source: "merged_rent_311.csv (all rows indexed by grid_id)",
    requested_grid_ids: gids,
    found_grid_ids: found.map((x) => x.grid_id),
    missing_grid_ids: missing,
    by_grid: found,
  };
  return `Direct grid-cell lookup from merged dataset:\n${JSON.stringify(payload, null, 2)}`;
}

async function loadContextData() {
  const [meta, geo, summary] = await Promise.all([fetchJson(DATA_META), fetchJson(DATA_GEOJSON), parseCsv(DATA_SUMMARY)]);
  state.meta = meta;
  state.geo = geo;
  state.summaryRows = summary;
  state.gridById.clear();
  state.summaryByCluster.clear();

  for (const row of summary) {
    state.summaryByCluster.set(String(row.cluster), row);
  }
  for (const feat of geo.features ?? []) {
    const id = String(feat?.properties?.grid_id ?? "");
    if (id) state.gridById.set(id, feat.properties);
  }
}

function normalizeSociopaperRaw(text) {
  return String(text || "")
    .replace(/-\r?\n/g, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n");
}

function buildSociopaperChunks(text) {
  const flat = normalizeSociopaperRaw(text).replace(/\n+/g, " ").replace(/\s+/g, " ").trim();
  if (!flat) return [];
  const sentences = flat.split(/(?<=[.!?])\s+/).filter(Boolean);
  const chunks = [];
  let buf = "";
  for (const s of sentences) {
    const next = buf ? `${buf} ${s}` : s;
    if (next.length >= SOCIOPAPER_CHUNK_TARGET && buf) {
      chunks.push(buf.trim());
      buf = s;
    } else {
      buf = next;
    }
  }
  if (buf.trim()) chunks.push(buf.trim());
  return chunks.map((t, i) => ({ id: i + 1, text: t }));
}

function sociopaperQueryTokens(query) {
  const raw = String(query || "")
    .toLowerCase()
    .match(/[a-z0-9]+/g);
  if (!raw) return [];
  return raw.filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

function sociopaperChunkScore(chunkText, tokens) {
  if (!tokens.length) return 0;
  const hay = chunkText.toLowerCase();
  let score = 0;
  for (const t of tokens) {
    if (hay.includes(t)) score += 1;
  }
  return score;
}

function retrieveSociopaperExcerpts(userQuery) {
  const tokens = sociopaperQueryTokens(userQuery);
  if (!state.sociopaperChunks.length) return "";
  const ranked = state.sociopaperChunks
    .map((c) => ({ c, score: sociopaperChunkScore(c.text, tokens) }))
    .sort((a, b) => b.score - a.score);
  const picked = (tokens.length ? ranked.filter((x) => x.score > 0) : ranked).slice(0, SOCIOPAPER_TOP_K).map((x) => x.c);
  const fallback = tokens.length && !picked.length ? ranked.slice(0, SOCIOPAPER_TOP_K).map((x) => x.c) : picked;
  const lines = fallback.map((c) => `[${SOCIOPAPER_ID} chunk ${c.id}]\n${c.text}`);
  return `Sociology paper excerpts (source: ${SOCIOPAPER_ID}.txt). Prefer these passages for concepts and citations when they are relevant; integrate with dashboard context where useful.\n\n${lines.join("\n\n")}`;
}

async function loadSociopaper() {
  try {
    const r = await fetch(SOCIOPAPER_TXT);
    if (!r.ok) throw new Error(`HTTP ${r.status} loading sociopaper`);
    const text = await r.text();
    state.sociopaperChunks = buildSociopaperChunks(text);
  } catch {
    state.sociopaperChunks = [];
  }
}

function contextPayload() {
  const { mode, clusterId, gridId } = currentContextSelection();
  const base = {
    mode,
    modeling_notes: {
      overall_fairness_level: "0-100 index for how balanced service access and quality are across areas",
      top3_gap_factors: "top 3 factors with the largest differences from city average in each cluster",
    },
    selected_cluster: clusterId,
  };

  if (mode === "global") {
    return {
      ...base,
      clusters: state.summaryRows,
      cluster_names: state.meta?.config?.cluster_names ?? {},
      supplemental_data: {
        grid_rent_311_rows: state.gridRent311Rows.length,
        grid_311_case_rows: state.caseSummaryByGrid.size,
      },
    };
  }

  if (mode === "cluster") {
    const c = String(clusterId);
    return {
      ...base,
      cluster: c,
      cluster_summary: state.summaryByCluster.get(c) ?? null,
      top_features: state.meta?.top3_features_per_cluster?.[c] ?? state.meta?.top3_features_per_cluster?.[Number(c)] ?? [],
      heuristics: state.meta?.heuristics?.[c] ?? state.meta?.heuristics?.[Number(c)] ?? null,
      supplemental_data: {
        grid_rent_311_rows: state.gridRent311Rows.length,
        cluster_case_summary: state.caseSummaryByCluster.get(c) ?? null,
      },
    };
  }

  const gid = String(gridId).trim();
  const row = state.gridById.get(gid) ?? null;
  return {
    ...base,
    grid_id: gid,
    grid_record: row,
    cluster_summary: row ? state.summaryByCluster.get(String(row.cluster)) ?? null : null,
    cluster_top_features: row
      ? state.meta?.top3_features_per_cluster?.[String(row.cluster)] ??
        state.meta?.top3_features_per_cluster?.[Number(row.cluster)] ??
        []
      : [],
    supplemental_data: {
      grid_rent_311: gid ? state.gridRentById.get(gid) ?? null : null,
      grid_311_case_summary: gid ? state.caseSummaryByGrid.get(gid) ?? null : null,
    },
  };
}

function renderContextPreview() {
  if (!els.contextPreview) return;
  els.contextPreview.value = JSON.stringify(contextPayload(), null, 2);
}

function bubble(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "assistant" ? "assistant" : "user"}`;
  const meta = document.createElement("div");
  meta.className = "msgMeta";
  meta.textContent = role === "assistant" ? "Assistant" : "You";
  const body = document.createElement("pre");
  body.className = "msgBody";
  body.textContent = role === "assistant" ? sanitizeAssistantText(text) : text;
  wrap.appendChild(meta);
  wrap.appendChild(body);
  return wrap;
}

function renderChat() {
  els.chatMessages.innerHTML = "";
  for (const m of state.chat) {
    els.chatMessages.appendChild(bubble(m.role, m.content));
  }
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function providerFromModel(model) {
  if (model.startsWith("gpt-")) return "openai";
  if (model.startsWith("claude-")) return "anthropic";
  if (model.startsWith("gemini-")) return "google";
  return "openai";
}

async function callOpenAI({ model, apiKey, messages, temperature, maxTokens }) {
  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
    }),
  });
  if (!r.ok) throw new Error(`OpenAI error ${r.status}: ${await r.text()}`);
  const data = await r.json();
  return data?.choices?.[0]?.message?.content ?? "(empty response)";
}

async function callAnthropic({ model, apiKey, messages, maxTokens, temperature }) {
  const system = messages.find((m) => m.role === "system")?.content ?? "";
  const msg = messages.filter((m) => m.role !== "system").map((m) => ({ role: m.role, content: m.content }));
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      system,
      messages: msg,
      max_tokens: maxTokens,
      temperature,
    }),
  });
  if (!r.ok) throw new Error(`Anthropic error ${r.status}: ${await r.text()}`);
  const data = await r.json();
  return data?.content?.map((c) => c?.text ?? "").join("\n").trim() || "(empty response)";
}

function toGeminiContents(messages) {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));
}

async function callGoogle({ model, apiKey, messages, temperature, maxTokens }) {
  const system = messages.find((m) => m.role === "system")?.content ?? "";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: system }] },
      contents: toGeminiContents(messages),
      generationConfig: {
        temperature,
        maxOutputTokens: maxTokens,
      },
    }),
  });
  if (!r.ok) throw new Error(`Google error ${r.status}: ${await r.text()}`);
  const data = await r.json();
  return data?.candidates?.[0]?.content?.parts?.map((p) => p?.text ?? "").join("\n").trim() || "(empty response)";
}

async function callModel(opts) {
  const provider = providerFromModel(opts.model);
  if (provider === "anthropic") return callAnthropic(opts);
  if (provider === "google") return callGoogle(opts);
  return callOpenAI(opts);
}

function sanitizeAssistantText(text) {
  // Light cleanup only. Do not globally replace words like "performance" or "z-score"—that breaks ML/STATS answers.
  let s = String(text || "")
    .replace(/\*\*"(.*?)"\*\*/g, "$1")
    .replace(/\*\*“(.*?)”\*\*/g, "$1")
    .replace(/\*\*'(.*?)'\*\*/g, "$1")
    .replace(/\*\*\s*["“'](.*?)["”']\s*\*\*/g, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/[“”"]/g, "");

  // Hide inline RAG citation markers in the UI, without collapsing line breaks.
  s = s.replace(/[ \t]*\[ref:\d+\][ \t]*/g, " ");
  s = s.replace(/[ \t]{2,}/g, " ");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

function fullSystemPrompt(userQuery) {
  const payloadText = JSON.stringify(contextPayload(), null, 2);
  const paper = retrieveSociopaperExcerpts(userQuery);
  const paperBlock = paper ? `\n\n${paper}` : "";
  const wantsRent = rentIntentFromQuery(userQuery);
  const wants311 = incidentIntentFromQuery(userQuery);
  const gridLookupExcerpt = retrieveGridCellDetails(userQuery);
  const rentExcerpt = wantsRent
    ? state.rentListings.length
      ? `\n\n${retrieveRentListings(userQuery)}`
      : "\n\nMerged rent+311 CSV is unavailable in outputs; answer using grid-level context only."
    : "";
  const incidentsExcerpt = wants311 && state.rentListings.length ? `\n\n${retrieve311Incidents(userQuery)}` : "";
  const { mode, clusterId, gridId } = currentContextSelection();
  const gridRentSummary = gridId ? state.gridRentById.get(gridId) ?? null : null;
  const gridCaseSummary = gridId ? state.caseSummaryByGrid.get(gridId) ?? null : null;
  const clusterCaseSummary = state.caseSummaryByCluster.get(clusterId) ?? null;
  const supplemental = [];
  supplemental.push(
    `Supplemental data availability: grid_level_rent_311 rows=${state.gridRent311Rows.length}, merged-grid 311 summaries=${state.caseSummaryByGrid.size}.`
  );
  if (mode === "grid" && gridId) {
    supplemental.push(`Grid-level enrichment for ${gridId}: ${JSON.stringify(gridRentSummary ?? {})}`);
    supplemental.push(`311 case summary for ${gridId}: ${JSON.stringify(gridCaseSummary ?? {})}`);
  } else if (mode === "cluster") {
    supplemental.push(`Cluster ${clusterId} 311 case summary: ${JSON.stringify(clusterCaseSummary ?? {})}`);
  }
  const supplementalBlock = `\n\nSupplemental CSV summaries:\n${supplemental.join("\n")}`;
  const gridCellBlock = gridLookupExcerpt ? `\n\n${gridLookupExcerpt}` : "";
  return `${els.systemPrompt.value.trim()}\n\n${RESPONSE_STYLE_GUARD}\n\nContext payload (JSON):\n${payloadText}${paperBlock}${rentExcerpt}${incidentsExcerpt}${gridCellBlock}${supplementalBlock}`;
}

async function send() {
  const userText = els.userInput.value.trim();
  if (!userText) return;
  const ragBase = (els.ragApiBase.value || "").trim();
  if (!ragBase) return setStatus("Missing RAG API base URL");
  window.RAG_API_BASE = ragBase;

  const urlParams = new URLSearchParams(location.search);
  const ragDebug = urlParams.get("ragDebug") === "1";

  const model = selectedModel();
  const temperature = Number(els.temperature.value || 0.3);
  const maxTokens = Number(els.maxTokens.value || 900);
  const topK = Number(els.topK.value || 8);

  state.chat.push({ role: "user", content: userText });
  state.chat = state.chat.slice(-MAX_HISTORY);
  els.userInput.value = "";
  renderChat();
  setStatus(`Calling RAG backend (${model})...`);
  saveState();

  try {
    const messages = [{ role: "system", content: fullSystemPrompt(userText) }, ...state.chat];
    const res = await ragChat({
      question: userText,
      messages,
      systemPrompt: fullSystemPrompt(userText),
      model,
      topK,
      temperature,
      maxTokens,
      debug: ragDebug,
    });
    const content = sanitizeAssistantText(String(res?.content ?? ""));
    const cites = Array.isArray(res?.citations) ? res.citations : [];
    const dbg = res?.retrievalDebug;
    const dbgBlock =
      ragDebug && dbg
        ? `\n\nRAG debug:\n- supabase: ${dbg.supabaseUrlHost || "(unknown)"}\n- key: ${dbg.keyType || "(unknown)"}\n- visible rows: ${
            typeof dbg.visibleRowCount === "number" ? dbg.visibleRowCount : "(unknown)"
          }\n- mode: ${dbg.retrievalMode || "(unknown)"}\n- titleHint: ${dbg.titleHint || "(none)"}`
        : "";
    // Hide sources list in the UI (citations still available in `res.citations` for debugging).
    void cites;
    const citeBlock = "";
    state.chat.push({ role: "assistant", content: `${content}${citeBlock}${dbgBlock}` });
    state.chat = state.chat.slice(-MAX_HISTORY);
    renderChat();
    setStatus(`Response received (${cites.length} citations)`);
    saveState();
  } catch (err) {
    const msg = String(err?.message || err);
    state.chat.push({ role: "assistant", content: `Error: ${msg}` });
    renderChat();
    setStatus("Request failed");
  }
}

function bindEvents() {
  els.floatingChatToggle?.addEventListener("click", () => {
    const dock = els.floatingChatDock;
    if (!dock) return;
    const collapsed = dock.classList.toggle("isCollapsed");
    els.floatingChatToggle.textContent = collapsed ? "+" : "−";
    els.floatingChatToggle.setAttribute("aria-expanded", String(!collapsed));
  });

  els.sendBtn.addEventListener("click", () => send());
  els.userInput.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.isComposing) return;
    if (e.shiftKey) return; // Shift+Enter inserts newline.
    e.preventDefault();
    send();
  });

  const persistHandlers = [
    els.modelSelect,
    els.customModel,
    els.apiKey,
    els.ragApiBase,
    els.temperature,
    els.maxTokens,
    els.topK,
    els.systemPrompt,
  ];
  for (const el of persistHandlers) {
    if (!el) continue;
    el.addEventListener("change", () => {
      renderContextPreview();
      saveState();
    });
  }

  els.reportCluster?.addEventListener("change", () => {
    renderContextPreview();
    saveState();
  });
  if (els.selGridId) {
    const observer = new MutationObserver(() => renderContextPreview());
    observer.observe(els.selGridId, { childList: true, characterData: true, subtree: true });
  }
  els.clearChat.addEventListener("click", () => {
    state.chat = [];
    renderChat();
    saveState();
    setStatus("Chat cleared");
  });
}

async function init() {
  els.systemPrompt.value = DEFAULT_SYSTEM_PROMPT;
  loadSavedState();
  bindEvents();
  renderChat();

  try {
    setStatus("Loading dashboard context and enrichment CSVs...");
    await Promise.all([loadContextData(), loadSociopaper()]);
    try {
      const merged = await loadRentDatasetWithFallback();
      state.rentListings = merged?.rentRows ?? [];
      state.mergedGridIndex = merged?.gridIndex ?? new Map();
      console.info(`rent dataset loaded from: ${merged?.sourceUrl || "(unknown)"}`);
    } catch (e) {
      state.rentListings = [];
      state.mergedGridIndex = new Map();
      console.warn("merged_rent_311 load failed", e);
    }
    try {
      state.gridRent311Rows = await loadGridRent311Dataset();
    } catch (e) {
      state.gridRent311Rows = [];
      console.warn("grid_level_rent_311 load failed", e);
    }
    reindexSupplementalData();
    renderContextPreview();
    const n = state.rentListings.length;
    const g = state.mergedGridIndex.size;
    const m = state.gridRent311Rows.length;
    const k = state.caseSummaryByGrid.size;
    setStatus(
      n || m || k || g
        ? `Ready (rent rows: ${n.toLocaleString()}, merged grid index: ${g.toLocaleString()}, grid enrichments: ${m.toLocaleString()}, merged 311 summaries: ${k.toLocaleString()})`
        : "Ready (supplemental CSVs unavailable)"
    );
  } catch (err) {
    const msg = String(err?.message || err);
    setStatus("Context load failed");
    els.contextPreview.value = `Failed to load outputs for context.\n${msg}`;
  }
}

init();
