import { ragChat } from "./rag_client.js";

// --- paths (module-relative) ---
const DATA_BASE = new URL("../outputs/", import.meta.url).href;
const DATA_RENT_CSV = new URL("rent_dataset_module2.csv", DATA_BASE).href;
const SOCIOPAPER_TXT = new URL("../sociopaper/zahnow1.txt", import.meta.url).href;

const STORAGE_KEY = "equity_public_chat_v1";
const MAX_HISTORY = 16;
const SOCIOPAPER_ID = "zahnow1";
const SOCIOPAPER_TOP_K = 5;
const SOCIOPAPER_CHUNK_TARGET = 1100;
const RENT_CONTEXT_MAX_ROWS = 22;
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
/** Canonical analysis_neighborhood values in rent_dataset_module2 (keep in sync with data). */
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



// --- DOM ---
const els = {
  floatingChatDock: document.getElementById("floatingPublicChat"),
  floatingChatBody: document.getElementById("floatingPublicChatBody"),
  floatingChatToggle: document.getElementById("publicChatToggle"),
  contextHint: document.getElementById("publicChatContextHint"),
  chatStatus: document.getElementById("publicChatStatus"),
  chatMessages: document.getElementById("publicChatMessages"),
  userInput: document.getElementById("publicChatInput"),
  sendBtn: document.getElementById("publicChatSend"),
  clearBtn: document.getElementById("publicChatClear"),
  selection: document.getElementById("selection"),
  selGridId: document.getElementById("selGridId"),
  reportCluster: document.getElementById("reportCluster"),
};

const state = {
  chat: /** @type {{role: "user"|"assistant"; content: string}[]} */ ([]),
  /** Filled from app.js (no second GeoJSON download) */
  meta: null,
  geo: null,
  summaryRows: [],
  gridById: new Map(),
  summaryByCluster: new Map(),

  /** @type {{ id: number; text: string }[]} */
  sociopaperChunks: [],
  /** @type {{ addr: string; hood: string; rent: number; beds: number | null; baths: number | null; sqft: number | null; year: number | null; district: number | null }[]} */
  rentListings: [],
  ready: false,
};

const DEFAULT_SYSTEM_PROMPT = `You are the assistant inside an urban service equity dashboard, but you behave like a normal helpful chat model: answer the user's actual question in a direct, natural tone.

Refusals (forbidden):
- Do NOT answer with stiff scope refusals such as "I can only help with urban service equity" or "ask me about urban services" when the user asks something else. Never deflect harmless test questions.

How to answer:
- Questions about this map, clusters, grids, fairness, services, or San Francisco housing in context: use the JSON context below; be concrete; say when something is uncertain.
- General questions (definitions, machine learning, statistics, what a paper says, etc.): answer straight. Use normal technical vocabulary when it helps. The server also attaches retrieved "Paper excerpts" with [ref:n] labels—use them when relevant and cite like [ref:1]; if excerpts are off-topic, say that in one short clause and answer from general knowledge.
- Optional sociology text (zahnow1 chunks) in the system message: use for social/urban questions when relevant; cite [zahnow1 chunk N]. Do not force paper citations for unrelated questions.

Style:
- Sound like a knowledgeable colleague, not a policy notice.
- Keep answers concise. Avoid crutch phrases ("I'm here to assist…").
- Do not use bold-quoted emphasis like **"..."**.`;

const RESPONSE_STYLE_GUARD = `Formatting:
- No bold-quoted emphasis patterns.

When (and only when) explaining THIS dashboard's JSON, map, cluster report, or SF housing-inventory rows for a public audience:
- Paraphrase internal metric names instead of reciting field names; avoid dumping raw JSON keys.
- Avoid naming schema-style cluster labels when plain language will do.
- If SF housing inventory rows are in context, use only those rows for address-level examples and note they are inventory records, not a live rental feed.

For questions that are not about this dashboard, ignore the paragraph above if it would block a clear answer.`;

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

  // Hide inline RAG citation markers in the public UI, without collapsing line breaks.
  s = s.replace(/[ \t]*\[ref:\d+\][ \t]*/g, " ");
  s = s.replace(/[ \t]{2,}/g, " ");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

function setStatus(msg) {
  if (els.chatStatus) els.chatStatus.textContent = msg;
}

function loadSavedState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    state.chat = Array.isArray(saved.chat)
      ? saved.chat.slice(-MAX_HISTORY).map((m) => ({
          ...m,
          content: m?.role === "assistant" ? sanitizeAssistantText(m?.content) : String(m?.content ?? ""),
        }))
      : [];
  } catch {
    // ignore
  }
}

function saveState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      chat: state.chat.slice(-MAX_HISTORY),
    })
  );
}

function isGridSelected() {
  if (!els.selection || !els.selGridId) return false;
  if (els.selection.classList.contains("hidden")) return false;
  const gid = String(els.selGridId.textContent || "").trim();
  return Boolean(gid && gid !== "—");
}

function reportClusterId() {
  return String(els.reportCluster?.value ?? "0");
}

function clusterName(c) {
  if (!state.meta?.config?.cluster_names) return `Cluster ${c}`;
  return state.meta.config.cluster_names[String(c)] ?? state.meta.config.cluster_names[c] ?? `Cluster ${c}`;
}

function updateContextHint() {
  if (!els.contextHint) return;
  if (!state.ready) {
    els.contextHint.textContent = "Loading data…";
    return;
  }
  if (isGridSelected()) {
    const gid = String(els.selGridId.textContent || "").trim();
    els.contextHint.textContent = `Focused on the selected area (${gid})`;
    return;
  }
  const c = reportClusterId();
  els.contextHint.textContent = `Focused on: ${clusterName(c)}`;
}

function reindexFromDashboard() {
  state.gridById.clear();
  state.summaryByCluster.clear();

  for (const row of state.summaryRows ?? []) {
    state.summaryByCluster.set(String(row.cluster), row);
  }
  for (const feat of state.geo?.features ?? []) {
    const id = String(feat?.properties?.grid_id ?? "");
    if (id) state.gridById.set(id, feat.properties);
  }
}

function contextPayload() {
  const base = {
    modeling_notes: {
      overall_fairness_level: "0-100 index for how balanced service access and quality are across areas",
      top3_gap_factors: "top 3 factors with the largest differences from city average in each cluster",
    },
    report_cluster_in_ui: reportClusterId(),
  };

  if (isGridSelected()) {
    const gid = String(els.selGridId.textContent || "").trim();
    const row = state.gridById.get(gid) ?? null;
    return {
      mode: "grid",
      ...base,
      grid_id: gid,
      grid_record: row,
      cluster_summary: row ? state.summaryByCluster.get(String(row.cluster)) ?? null : null,
      cluster_top_features: row
        ? state.meta?.top3_features_per_cluster?.[String(row.cluster)] ??
          state.meta?.top3_features_per_cluster?.[Number(row.cluster)] ??
          []
        : [],
    };
  }

  const c = reportClusterId();
  return {
    mode: "cluster",
    ...base,
    cluster: c,
    cluster_summary: state.summaryByCluster.get(c) ?? null,
    top_features: state.meta?.top3_features_per_cluster?.[c] ?? state.meta?.top3_features_per_cluster?.[Number(c)] ?? [],
    heuristics: state.meta?.heuristics?.[c] ?? state.meta?.heuristics?.[Number(c)] ?? null,
  };
}

function loadRentDataset() {
  return new Promise((resolve, reject) => {
    const rows = [];
    Papa.parse(DATA_RENT_CSV, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      step: (res) => {
        const r = res.data;
        if (!r || typeof r !== "object") return;
        const rentRaw = r.monthly_rent_clean;
        const rent =
          typeof rentRaw === "number" && !Number.isNaN(rentRaw)
            ? rentRaw
            : Number(String(rentRaw ?? "").replace(/,/g, "").trim());
        if (!Number.isFinite(rent) || rent <= 0) return;
        const addr = String(r.block_address ?? "").trim();
        const hood = String(r.analysis_neighborhood ?? "").trim();
        if (!addr || !hood) return;
        rows.push({
          addr,
          hood,
          rent,
          beds: typeof r.bedrooms_clean === "number" && !Number.isNaN(r.bedrooms_clean) ? r.bedrooms_clean : null,
          baths: typeof r.bathrooms_clean === "number" && !Number.isNaN(r.bathrooms_clean) ? r.bathrooms_clean : null,
          sqft: typeof r.sqft_avg === "number" && !Number.isNaN(r.sqft_avg) ? r.sqft_avg : null,
          year: typeof r.submission_year === "number" && !Number.isNaN(r.submission_year) ? r.submission_year : null,
          district: typeof r.supervisor_district === "number" && !Number.isNaN(r.supervisor_district) ? r.supervisor_district : null,
        });
      },
      complete: () => resolve(rows),
      error: (err) => reject(err),
    });
  });
}

function rentIntentFromQuery(q) {
  const s = String(q || "").toLowerCase();
  if (!s.trim()) return false;
  const hasMoney =
    /\$\s*\d/.test(s) ||
    /\d{3,5}\s*[-–—to]+\s*\d{3,5}/.test(s) ||
    /\b(?:under|below|over|above|less than|more than|at least)\s*\$?\s*\d{3,5}\b/.test(s);
  const housingWords =
    /\b(rent|rental|apartment|housing|lease|landlord|tenant|bedroom|bedrooms|studio|sqft|sq\.?\s*ft|afford|move|live|neighborhood|quiet|location|place|options|listing|unit|flat)\b/.test(
      s,
    );
  if (/\bsf\b|\bsan francisco\b|\bay area\b/.test(s) && housingWords) return true;
  if (housingWords && hasMoney) return true;
  if (hasMoney && /\b(live|place|quiet|budget|ideal|looking|want)\b/.test(s)) return true;
  if (/\b(rent|rental|apartment|housing|lease|bedroom|studio)\b/.test(s) && hasMoney) return true;
  if (/\$\s*\d{3,5}\b/.test(s) && /\b(rent|month|budget|afford)\b/.test(s)) return true;
  return false;
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
    source: "San Francisco Housing Inventory (rent_dataset_module2.csv)",
    note: "Rows are city-reported units; not guaranteed to be vacant or on the market today.",
    filters_applied: { min, max, neighborhoods: hoods, bedroom_hint: bedHint, quiet_bias: wantQuiet },
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



function buildFullSystemPrompt(userQuery) {
  const payloadText = JSON.stringify(contextPayload(), null, 2);
  const paper = retrieveSociopaperExcerpts(userQuery);
  const paperBlock = paper ? `\n\n${paper}` : "";
  const rentExcerpt = rentIntentFromQuery(userQuery)
    ? state.rentListings.length
      ? `\n\n${retrieveRentListings(userQuery)}`
      : "\n\nSan Francisco housing inventory CSV was not loaded; answer without inventing specific addresses."
    : "";
  return `${DEFAULT_SYSTEM_PROMPT.trim()}\n\n${RESPONSE_STYLE_GUARD}\n\nContext payload (JSON):\n${payloadText}${paperBlock}${rentExcerpt}`;
}

function bubble(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "assistant" ? "assistant" : "user"}`;
  const meta = document.createElement("div");
  meta.className = "msgMeta";
  meta.textContent = role === "assistant" ? "Assistant" : "You";
  const body = document.createElement("div");
  body.className = "msgBody";
  body.textContent = role === "assistant" ? sanitizeAssistantText(text) : text;
  wrap.appendChild(meta);
  wrap.appendChild(body);
  return wrap;
}

function renderChat() {
  if (!els.chatMessages) return;
  els.chatMessages.innerHTML = "";
  for (const m of state.chat) {
    els.chatMessages.appendChild(bubble(m.role, m.content));
  }
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function send() {
  const userText = String(els.userInput?.value || "").trim();
  if (!userText) return;
  if (!state.ready) return setStatus("Still loading dashboard data…");

  const model = "gpt-4o";
  const temperature = 0.5;
  const maxTokens = 900;
  const topK = 8;

  state.chat.push({ role: "user", content: userText });
  state.chat = state.chat.slice(-MAX_HISTORY);
  els.userInput.value = "";
  renderChat();
  setStatus("Thinking…");
  saveState();

  try {
    const messages = [{ role: "system", content: buildFullSystemPrompt(userText) }, ...state.chat];
    const res = await ragChat({
      question: userText,
      messages,
      systemPrompt: buildFullSystemPrompt(userText),
      model,
      topK,
      temperature,
      maxTokens,
    });
    const content = sanitizeAssistantText(String(res?.content ?? ""));
    const cites = Array.isArray(res?.citations) ? res.citations : [];
    // Public chat: do not show sources/citations to end users.
    void cites;
    state.chat.push({ role: "assistant", content });
    state.chat = state.chat.slice(-MAX_HISTORY);
    renderChat();
    setStatus(cites.length ? `Ready (used ${cites.length} paper excerpts)` : "Ready");
    saveState();
  } catch (err) {
    const msg = String(err?.message || err);
    state.chat.push({ role: "assistant", content: `Sorry — I could not get an answer.\n\n${msg}` });
    state.chat = state.chat.slice(-MAX_HISTORY);
    renderChat();
    setStatus("Error");
    saveState();
  }
}

function clearChat() {
  state.chat = [];
  renderChat();
  saveState();
  setStatus("Cleared");
}

async function onDashboardReady(detail) {
  state.meta = detail.meta;
  state.geo = detail.geo;
  state.summaryRows = detail.summaryRows;
  reindexFromDashboard();
  setStatus("Loading local references…");

  try {
    await Promise.all([loadSociopaper(), loadRentDataset().then((rows) => (state.rentListings = rows))]);
  } catch {
    // non-fatal
  }

  state.ready = true;
  loadSavedState();
  renderChat();
  updateContextHint();
  setStatus("Ready");
}

function bindEvents() {
  els.floatingChatToggle?.addEventListener("click", () => {
    const dock = els.floatingChatDock;
    if (!dock) return;
    const collapsed = dock.classList.toggle("isCollapsed");
    els.floatingChatToggle.textContent = collapsed ? "+" : "−";
    els.floatingChatToggle.setAttribute("aria-expanded", String(!collapsed));
  });

  els.sendBtn?.addEventListener("click", () => send());
  els.clearBtn?.addEventListener("click", () => clearChat());
  els.userInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  window.addEventListener("equity-dashboard-ready", (e) => {
    onDashboardReady(e.detail).catch((err) => {
      console.error(err);
      setStatus("Failed to initialize chat");
    });
  });
  window.addEventListener("equity-selection-changed", () => updateContextHint());
}

bindEvents();
updateContextHint();
setStatus("Loading…");
