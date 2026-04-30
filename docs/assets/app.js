import { CLUSTER_COLORS, equityColor, clamp, INDICATOR_LABELS, fmt } from "./utils.js";

const MODE_CLUSTER = "cluster";
const MODE_EQUITY = "equity";
const MODE_LISA = "lisa";

const LISA_COLORS = {
  LL: "#dc2626",
  LH: "#f59e0b",
  HH: "#16a34a",
  HL: "#3b82f6",
  NS: "#475569",
};
const LISA_LABELS = {
  LL: "Low-Low (underserved cluster)",
  LH: "Low-High (struggling pocket)",
  HH: "High-High (well-served cluster)",
  HL: "High-Low (well-served outlier)",
  NS: "Not significant",
};

// Resolve from this module so paths work on GitHub Pages, local /docs server, or nested URLs.
const DATA_BASE = new URL("../outputs/", import.meta.url);
const DATA_GEOJSON = new URL("grid_points.geojson", DATA_BASE).href;
const DATA_META = new URL("metadata.json", DATA_BASE).href;
const DATA_SUMMARY = new URL("cluster_summary.csv", DATA_BASE).href;
const DATA_Z = new URL("cluster_feature_zscores.csv", DATA_BASE).href;
const DATA_POINT_ADVICE = new URL("grid_point_advice.json", DATA_BASE).href;
const DATA_NBHD = new URL("sf_neighborhoods.geojson", DATA_BASE).href;

const els = {
  colorMode: document.getElementById("colorMode"),
  clusterFilter: document.getElementById("clusterFilter"),
  equityMin: document.getElementById("equityMin"),
  equityMax: document.getElementById("equityMax"),
  applyFilters: document.getElementById("applyFilters"),
  legend: document.getElementById("legend"),
  dataPath: document.getElementById("dataPath"),
  selPanelTitle: document.getElementById("selPanelTitle"),
  selectionEmpty: document.getElementById("selectionEmpty"),
  selection: document.getElementById("selection"),
  selGridId: document.getElementById("selGridId"),
  selClusterSection: document.getElementById("selClusterSection"),
  selCluster: document.getElementById("selCluster"),
  selClusterN: document.getElementById("selClusterN"),
  selClusterNeighborhood: document.getElementById("selClusterNeighborhood"),
  selClusterEquityMean: document.getElementById("selClusterEquityMean"),
  selClusterTop: document.getElementById("selClusterTop"),
  selEquitySection: document.getElementById("selEquitySection"),
  selEquity: document.getElementById("selEquity"),
  selEquityNeighborhood: document.getElementById("selEquityNeighborhood"),
  selTop: document.getElementById("selTop"),
  selLisaSection: document.getElementById("selLisaSection"),
  selLisaQuadrant: document.getElementById("selLisaQuadrant"),
  selLisaEquity: document.getElementById("selLisaEquity"),
  selLisaNeighborhood: document.getElementById("selLisaNeighborhood"),
  selLisaI: document.getElementById("selLisaI"),
  selLisaP: document.getElementById("selLisaP"),
  clusterLink: document.getElementById("clusterLink"),
  clearSelection: document.getElementById("clearSelection"),
  reportCluster: document.getElementById("reportCluster"),
  reportAnchor: document.getElementById("report"),
  clusterName: document.getElementById("clusterName"),
  statN: document.getElementById("statN"),
  statEquityMean: document.getElementById("statEquityMean"),
  statEquityMedian: document.getElementById("statEquityMedian"),
  statEquityBand: document.getElementById("statEquityBand"),
  statPerf: document.getElementById("statPerf"),
  statNeed: document.getElementById("statNeed"),
  direNeeds: document.getElementById("direNeeds"),
  priorityQueue: document.getElementById("priorityQueue"),
  needsAndInterventions: document.getElementById("needsAndInterventions"),
  pcaS: document.getElementById("pcaS"),
  pcaN: document.getElementById("pcaN"),
  pointAdviceCompact: document.getElementById("pointAdviceCompact"),
  pointAdviceCompactText: document.getElementById("pointAdviceCompactText"),
  pointNeedPanelMeta: document.getElementById("pointNeedPanelMeta"),
  pointDireNeeds: document.getElementById("pointDireNeeds"),
  pointQueue: document.getElementById("pointQueue"),
};

if (els.dataPath) els.dataPath.textContent = "outputs/grid_points.geojson";

let meta = null;
let map = null;
let layer = null;
let nbhdLayer = null;
let nbhdGeo = null;
let geo = null;
let summaryRows = [];
let zRows = [];
let zChart = null;
let sortedScores = [];
const EQUITY_HIST_BINS_MAX = 10;

/** @type {Record<string, any> | null} */
let pointAdviceByGrid = null;
/** @type {object | null} */
let selectedPointProps = null;

// Chart.js inline plugins from Kai's version.
const zeroLinePlugin = {
  id: "zeroLine",
  afterDraw(chart) {
    const {
      ctx,
      scales: { x, y },
    } = chart;
    const xPos = x.getPixelForValue(0);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(xPos, y.top);
    ctx.lineTo(xPos, y.bottom);
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  },
};

const barValuePlugin = {
  id: "barValue",
  afterDatasetsDraw(chart) {
    const {
      ctx,
      scales: { x },
    } = chart;
    const zero = x.getPixelForValue(0);
    chart.data.datasets.forEach((dataset, i) => {
      chart.getDatasetMeta(i).data.forEach((bar, index) => {
        const value = dataset.data[index];
        if (value === undefined || value === null) return;
        const label = `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
        ctx.save();
        ctx.fillStyle = "rgba(232,234,240,0.9)";
        ctx.font = "bold 10px ui-sans-serif,system-ui,sans-serif";
        ctx.textBaseline = "middle";
        if (value >= 0) {
          ctx.textAlign = "left";
          ctx.fillText(label, Math.max(bar.x, zero) + 4, bar.y);
        } else {
          ctx.textAlign = "right";
          ctx.fillText(label, Math.min(bar.x, zero) - 4, bar.y);
        }
        ctx.restore();
      });
    });
  },
};

function clusterName(c) {
  if (!meta?.config?.cluster_names) return `Cluster ${c}`;
  return meta.config.cluster_names[String(c)] ?? meta.config.cluster_names[c] ?? `Cluster ${c}`;
}

function featureName(code) {
  return INDICATOR_LABELS[code] ?? code;
}

function humanizeTopFeatures(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return "—";
  const names = text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((code) => featureName(code));
  return names.length ? names.join(", ") : text;
}

function selectedEquityRange() {
  const a = clamp(Number(els.equityMin?.value ?? 0), 0, 100);
  const b = clamp(Number(els.equityMax?.value ?? 100), 0, 100);
  return [Math.min(a, b), Math.max(a, b)];
}

function rawToPercent(score) {
  const n = sortedScores.length;
  if (!n || !Number.isFinite(score)) return null;
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedScores[mid] < score) lo = mid + 1;
    else hi = mid;
  }
  return Math.min(99, Math.floor((lo / n) * 100));
}

function passesFilters(props) {
  const cf = els.clusterFilter?.value ?? "all";
  if (cf !== "all" && String(props.cluster) !== cf) return false;
  const [emin, emax] = selectedEquityRange();
  const pct = rawToPercent(Number(props.equity_score));
  if (Number.isFinite(pct) && (pct < emin || pct > emax)) return false;
  return true;
}

function formatTop3(zRow) {
  if (!zRow) return "—";
  const lines = Object.entries(zRow)
    .filter(([k]) => k !== "cluster")
    .map(([k, v]) => ({ k, z: Number(v) }))
    .filter((d) => Number.isFinite(d.z))
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
    .slice(0, 3)
    .map((d) => `${INDICATOR_LABELS[d.k] ?? d.k} (${d.z >= 0 ? "+" : ""}${d.z.toFixed(2)})`);
  return lines.length ? lines.join("<br>") : "—";
}

function markerStyle(props) {
  const mode = els.colorMode?.value ?? MODE_EQUITY;
  if (mode === MODE_CLUSTER) {
    const c = CLUSTER_COLORS[props.cluster] ?? "#888";
    return { color: c, fillColor: c };
  }
  if (mode === MODE_LISA) {
    const q = props.lisa_quadrant ?? "NS";
    const c = LISA_COLORS[q] ?? LISA_COLORS.NS;
    return { color: c, fillColor: c };
  }
  const pct = rawToPercent(Number(props.equity_score));
  const [emin, emax] = selectedEquityRange();
  const span = Math.max(1e-9, emax - emin);
  const c = equityColor(clamp(((pct ?? 0) - emin) / span, 0, 1));
  return { color: c, fillColor: c };
}

function fmtRawEdge(v, span) {
  if (span <= 0.1) return v.toFixed(4);
  if (span <= 1) return v.toFixed(3);
  if (span <= 10) return v.toFixed(2);
  if (span <= 100) return v.toFixed(1);
  return v.toFixed(0);
}

function renderEquityHistogram() {
  if (!geo?.features?.length) {
    return `<div class="legendHint">Distribution loading...</div>`;
  }
  const [emin, emax] = selectedEquityRange();

  // Collect ALL raw scores to determine the full data range.
  const allScores = [];
  for (const f of geo.features) {
    const v = Number(f?.properties?.equity_score);
    if (Number.isFinite(v)) allScores.push(v);
  }
  if (!allScores.length) {
    return `<div class="legendHint">No equity scores available.</div>`;
  }

  const globalMin = Math.min(...allScores);
  const globalMax = Math.max(...allScores);

  // Map percentile filter limits back to raw score boundaries.
  allScores.sort((a, b) => a - b);
  const n = allScores.length;
  const loIdx = Math.min(n - 1, Math.max(0, Math.floor((emin / 100) * n)));
  const hiIdx = Math.min(n - 1, Math.max(0, Math.ceil((emax / 100) * n) - 1));
  const rawLo = allScores[loIdx];
  const rawHi = allScores[hiIdx];

  // Collect visible values within the percentile filter.
  const values = [];
  for (const f of geo.features) {
    const v = Number(f?.properties?.equity_score);
    if (!Number.isFinite(v)) continue;
    const pct = rawToPercent(v);
    if (!Number.isFinite(pct) || pct < emin || pct > emax) continue;
    values.push(v);
  }
  if (!values.length) {
    return `<div class="legendHint">No equity scores in percentile ${emin.toFixed(0)}–${emax.toFixed(0)}.</div>`;
  }

  // Build bins spanning rawLo → rawHi (the filtered raw range).
  const span = Math.max(1e-9, rawHi - rawLo);
  const bins = EQUITY_HIST_BINS_MAX;
  const counts = Array.from({ length: bins }, () => 0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor(((v - rawLo) / span) * bins));
    counts[idx] += 1;
  }

  const maxCount = Math.max(...counts, 1);
  const bars = counts
    .map((count, i) => {
      const lo = rawLo + i * (span / bins);
      const hi = rawLo + (i + 1) * (span / bins);
      const h = Math.max(4, Math.round((count / maxCount) * 36));
      return `<div class="histBar" style="height:${h}px" title="${fmtRawEdge(lo, span)}–${fmtRawEdge(hi, span)}: ${count}"></div>`;
    })
    .join("");

  const catRows = counts
    .map((count, i) => {
      const lo = rawLo + i * (span / bins);
      const hi = rawLo + (i + 1) * (span / bins);
      return `<div class="histCat"><span>${fmtRawEdge(lo, span)}–${fmtRawEdge(hi, span)}</span><b>${count.toLocaleString()}</b></div>`;
    })
    .join("");

  return `
    <div class="histWrap">
      <div class="histHeader">
        <span>Distribution (raw equity score)</span>
        <span>n = ${values.length.toLocaleString()}</span>
      </div>
      <div class="histBars">${bars}</div>
      <div class="histLabels">
        <span>${fmtRawEdge(rawLo, span)}</span>
        <span>${fmtRawEdge((rawLo + rawHi) / 2, span)}</span>
        <span>${fmtRawEdge(rawHi, span)}</span>
      </div>
      <div class="histCats">${catRows}</div>
    </div>
  `;
}

function renderLegend() {
  if (!els.legend) return;
  const mode = els.colorMode?.value ?? MODE_EQUITY;
  if (mode === MODE_CLUSTER) {
    els.legend.innerHTML = `
      <div class="legendTitle">Legend: Cluster</div>
      ${[0, 1, 2, 3]
        .map(
          (c) => `
        <div class="legendRow">
          <div class="swatch" style="background:${CLUSTER_COLORS[c]}"></div>
          <div>${clusterName(c)}</div>
        </div>`
        )
        .join("")}
    `;
    return;
  }
  if (mode === MODE_LISA) {
    const counts = { LL: 0, LH: 0, HH: 0, HL: 0, NS: 0 };
    for (const f of geo?.features ?? []) {
      const q = f.properties?.lisa_quadrant ?? "NS";
      counts[q] = (counts[q] ?? 0) + 1;
    }
    const order = ["LL", "LH", "HH", "HL", "NS"];
    els.legend.innerHTML = `
      <div class="legendTitle">Legend: LISA quadrant</div>
      ${order
        .map(
          (q) => `
        <div class="legendRow">
          <div class="swatch" style="background:${LISA_COLORS[q]}"></div>
          <div style="flex:1">${LISA_LABELS[q]}</div>
          <div style="color:var(--muted);font-size:11px">${counts[q].toLocaleString()}</div>
        </div>`
        )
        .join("")}
      <div class="legendHint">Significance: p &le; 0.05 (KNN k=8, 999 perms)</div>
    `;
    return;
  }
  const [emin, emax] = selectedEquityRange();
  els.legend.innerHTML = `
    <div class="legendTitle">Legend: Equity score (%)</div>
    <div class="ramp"></div>
    <div class="rampLabels"><span>${emin.toFixed(0)}</span><span>${((emin + emax) / 2).toFixed(0)}</span><span>${emax.toFixed(0)}</span></div>
    ${renderEquityHistogram()}
    <div class="legendHint">Red = lower (within selected range)</div>
  `;
}

function show(el, on = true) {
  if (!el) return;
  el.classList.toggle("hidden", !on);
}

function clearSelection() {
  selectedPointProps = null;
  show(els.selectionEmpty, true);
  show(els.selection, false);
  renderPointLevelAdvice();
  window.dispatchEvent(new CustomEvent("equity-selection-changed"));
}

function setSelection(props) {
  selectedPointProps = props;
  show(els.selectionEmpty, false);
  show(els.selection, true);
  if (els.selGridId) els.selGridId.textContent = props.grid_id ?? "—";

  const mode = els.colorMode?.value ?? MODE_EQUITY;
  const zRow = zRows.find((r) => String(r.cluster) === String(props.cluster));
  const neighborhood = props.neighborhood && String(props.neighborhood).trim() ? String(props.neighborhood) : "—";

  show(els.selClusterSection, false);
  show(els.selEquitySection, false);
  show(els.selLisaSection, false);

  if (mode === MODE_CLUSTER && els.selClusterSection) {
    if (els.selPanelTitle) els.selPanelTitle.textContent = "Cluster Report";
    show(els.selClusterSection, true);
    if (els.selCluster) els.selCluster.textContent = clusterName(props.cluster);
    const row = summaryRows.find((r) => String(r.cluster) === String(props.cluster));
    if (els.selClusterN) els.selClusterN.textContent = row?.n_grids_scored?.toLocaleString?.() ?? "—";
    if (els.selClusterNeighborhood) els.selClusterNeighborhood.textContent = neighborhood;
    if (els.selClusterEquityMean) els.selClusterEquityMean.textContent = row ? fmt(row.equity_mean, 2) : "—";
    if (els.selClusterTop) els.selClusterTop.innerHTML = formatTop3(zRow);
    if (els.clusterLink) {
      els.clusterLink.textContent = "View Cluster Report";
      els.clusterLink.href = "#clusterReportPanel";
    }
    setReportCluster(props.cluster);
  } else if (mode === MODE_LISA && els.selLisaSection) {
    if (els.selPanelTitle) els.selPanelTitle.textContent = "LISA Quadrant Report";
    show(els.selLisaSection, true);
    const q = props.lisa_quadrant ?? "NS";
    if (els.selLisaQuadrant) {
      els.selLisaQuadrant.textContent = LISA_LABELS[q] ?? q;
      const qc = LISA_COLORS[q] ?? LISA_COLORS.NS;
      els.selLisaQuadrant.style.background = `${qc}33`;
      els.selLisaQuadrant.style.borderColor = qc;
      els.selLisaQuadrant.style.color = qc;
    }
    const raw = Number(props.equity_score);
    if (els.selLisaEquity) els.selLisaEquity.textContent = Number.isFinite(raw) ? raw.toFixed(2) : "—";
    if (els.selLisaNeighborhood) els.selLisaNeighborhood.textContent = neighborhood;
    if (els.selLisaI) {
      const li = Number(props.lisa_I);
      els.selLisaI.textContent = Number.isFinite(li) ? li.toFixed(3) : "—";
    }
    if (els.selLisaP) {
      const lp = Number(props.lisa_p);
      els.selLisaP.textContent = Number.isFinite(lp) ? lp.toFixed(3) : "—";
    }
    if (els.clusterLink) {
      els.clusterLink.textContent = "View LISA Quadrant Report";
      els.clusterLink.href = "#lisaGroupsPanel";
    }
  } else {
    if (els.selPanelTitle) els.selPanelTitle.textContent = "Equity Score Report";
    if (els.selEquitySection) {
      show(els.selEquitySection, true);
      const raw = Number(props.equity_score);
      const pct = rawToPercent(raw);
      if (els.selEquity) {
        els.selEquity.textContent = Number.isFinite(raw) ? `${raw.toFixed(2)}${pct !== null ? ` (${pct}th pctile)` : ""}` : "—";
      }
      if (els.selEquityNeighborhood) els.selEquityNeighborhood.textContent = neighborhood;
      if (els.selTop) els.selTop.innerHTML = formatTop3(zRow);
      if (els.clusterLink) {
        els.clusterLink.textContent = "View Equity Score Report";
        els.clusterLink.href = "#lowEquityPanel";
      }
    } else {
      // Backward compatibility for Clara's compact side panel.
      if (els.selCluster) els.selCluster.textContent = clusterName(props.cluster);
      const raw = Number(props.equity_score);
      const pct = rawToPercent(raw);
      if (els.selEquity) {
        els.selEquity.textContent = Number.isFinite(raw) ? `${raw.toFixed(2)}${pct !== null ? ` (${pct}th pctile)` : ""}` : "—";
      }
      if (els.selTop) els.selTop.textContent = humanizeTopFeatures(props.top3_features);
      if (els.clusterLink) {
        els.clusterLink.textContent = "View Equity Score Report";
        els.clusterLink.href = "#lowEquityPanel";
      }
    }
  }

  renderPointLevelAdvice();
  window.dispatchEvent(new CustomEvent("equity-selection-changed"));
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

function renderPca() {
  const p = meta?.pca_weights?.service_performance ?? {};
  const n = meta?.pca_weights?.service_need ?? {};
  if (els.pcaS) els.pcaS.textContent = JSON.stringify(p, null, 2);
  if (els.pcaN) els.pcaN.textContent = JSON.stringify(n, null, 2);
}

function renderSummary(c) {
  const row = summaryRows.find((r) => String(r.cluster) === String(c));
  if (els.clusterName) els.clusterName.textContent = clusterName(c);
  if (!row) return;
  if (els.statN) els.statN.textContent = row.n_grids_scored?.toLocaleString?.() ?? String(row.n_grids_scored ?? "—");
  if (els.statEquityMean) els.statEquityMean.textContent = fmt(row.equity_mean, 2);
  if (els.statEquityMedian) els.statEquityMedian.textContent = fmt(row.equity_median, 2);
  if (els.statEquityBand) els.statEquityBand.textContent = `${fmt(row.equity_p10, 2)} → ${fmt(row.equity_p90, 2)}`;
  if (els.statPerf) els.statPerf.textContent = fmt(row.performance_mean, 2);
  if (els.statNeed) els.statNeed.textContent = fmt(row.need_mean, 2);
}

function renderZChart(c) {
  const row = zRows.find((r) => String(r.cluster) === String(c));
  if (!row) return;
  const items = Object.keys(row)
    .filter((k) => k !== "cluster" && row[k] !== null && row[k] !== undefined && !Number.isNaN(row[k]))
    .map((k) => ({ k, z: Number(row[k]) }))
    .filter((d) => Number.isFinite(d.z))
    .sort((a, b) => a.z - b.z);

  const labels = items.map((d) => INDICATOR_LABELS[d.k] ?? d.k);
  const data = items.map((d) => d.z);
  const colors = items.map((d) => (d.z >= 0 ? "rgba(34,197,94,.55)" : "rgba(239,68,68,.55)"));
  const borders = items.map((d) => (d.z >= 0 ? "rgba(34,197,94,1)" : "rgba(239,68,68,1)"));

  const ctx = document.getElementById("zChart");
  if (!ctx) return;
  if (zChart) zChart.destroy();
  zChart = new Chart(ctx, {
    type: "bar",
    plugins: [zeroLinePlugin, barValuePlugin],
    data: {
      labels,
      datasets: [
        {
          label: "z-score vs city avg",
          data,
          backgroundColor: colors,
          borderColor: borders,
          borderWidth: 1,
          borderRadius: 3,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items_) => items_[0].label,
            label: (item) => {
              const v = item.parsed.x;
              const dir = v >= 0 ? "above" : "below";
              return ` ${v >= 0 ? "+" : ""}${v.toFixed(3)} sigma ${dir} city average`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,.08)" },
          ticks: {
            color: "rgba(232,234,240,.75)",
            callback: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}sigma`,
          },
          title: {
            display: true,
            text: "Standard deviations from city average",
            color: "rgba(232,234,240,.5)",
            font: { size: 11 },
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: "rgba(232,234,240,.9)", font: { size: 11 } },
        },
      },
    },
  });
}

function renderHeuristics(c, target = null) {
  const tgtNeeds = target ?? els.direNeeds;
  const h = meta?.heuristics?.[String(c)] ?? meta?.heuristics?.[c];
  if (!h) {
    if (target) target.textContent = "—";
    else {
      if (els.direNeeds) els.direNeeds.textContent = "—";
      if (els.priorityQueue) els.priorityQueue.textContent = "—";
    }
    return;
  }

  const priClass = (p) => (p || "").toLowerCase();
  if (tgtNeeds) {
    tgtNeeds.innerHTML = (h.needs ?? [])
      .map((n) => {
        const actions = (n.actions ?? []).map((a) => `<li>${a}</li>`).join("");
        return `
          <div class="needCard">
            <div class="pill ${priClass(n.priority)}">${n.priority} · #${n.rank}</div>
            <div class="needTitle">${n.title}</div>
            <div class="needDesc">${n.desc}</div>
            <ul class="smallNote" style="margin:0;padding-left:18px">${actions}</ul>
          </div>
        `;
      })
      .join("");
  }

  if (!target && els.priorityQueue) {
    els.priorityQueue.innerHTML = (h.queue ?? [])
      .map(([num, action, why]) => {
        return `
          <div class="queueItem">
            <div class="queueNum">${num}</div>
            <div>
              <div class="queueAction">${action}</div>
              <div class="queueWhy">→ ${why}</div>
            </div>
          </div>
        `;
      })
      .join("");
  }
}

function pointPriClass(p) {
  const x = (p || "").toLowerCase();
  if (x === "info") return "info";
  return x;
}

function renderPointLevelAdvice() {
  const setEmpty = (msg) => {
    if (els.pointDireNeeds) els.pointDireNeeds.textContent = "—";
    if (els.pointQueue) els.pointQueue.textContent = "—";
    if (els.pointNeedPanelMeta) els.pointNeedPanelMeta.textContent = msg;
    if (els.pointAdviceCompact) els.pointAdviceCompact.classList.add("hidden");
  };

  if (!selectedPointProps || !selectedPointProps.grid_id) {
    setEmpty("Select a point on the map for grid-level needs and actions.");
    return;
  }
  if (pointAdviceByGrid == null) {
    setEmpty("Point advice file not found (re-run the pipeline to generate grid_point_advice.json).");
    return;
  }
  if (Object.keys(pointAdviceByGrid).length === 0) {
    setEmpty("Point advice is empty. Run the pipeline (it writes grid_point_advice.json from your grid scores and optional data/*.csv files).");
    return;
  }

  const gid = String(selectedPointProps.grid_id);
  const block = pointAdviceByGrid[gid] ?? null;
  if (!block) {
    setEmpty(`No point-level advice for grid ${gid} (re-run the pipeline with the same input grid set).`);
    return;
  }

  const needs = (block.needs ?? []).filter((n) => n && n.title);
  const visibleNeeds = needs.filter((n) => n.feature !== "file_context");
  const sols = block.solutions ?? [];

  if (els.pointNeedPanelMeta) {
    const feat = (visibleNeeds.find((n) => n.feature) || {}).feature;
    els.pointNeedPanelMeta.textContent = `Grid ${gid}` + (feat ? ` · strongest signal: ${featureName(feat)}` : "");
  }

  if (els.pointDireNeeds) {
    if (!visibleNeeds.length) {
      els.pointDireNeeds.textContent = "—";
    } else {
      els.pointDireNeeds.innerHTML = visibleNeeds
        .map((n) => {
          const tag = `${INDICATOR_LABELS[n.feature] || n.feature} (z ${n.z})`;
          return `
            <div class="needCard">
              <div class="pill ${pointPriClass(n.priority)}">${n.priority} · ${tag}</div>
              <div class="needTitle">${n.title}</div>
              <div class="needDesc">${n.desc}</div>
            </div>
          `;
        })
        .join("");
    }
  }

  if (els.pointQueue) {
    if (!sols.length) {
      els.pointQueue.textContent = "—";
    } else {
      els.pointQueue.innerHTML = sols
        .map(
          (s, i) => `
          <div class="queueItem">
            <div class="queueNum">${String(i + 1).padStart(2, "0")}</div>
            <div>
              <div class="queueAction">${s}</div>
              <div class="queueWhy">Point-level action (from feature templates)</div>
            </div>
          </div>
        `
        )
        .join("");
    }
  }

  if (els.pointAdviceCompact && els.pointAdviceCompactText) {
    const dataNeeds = needs.filter((n) => n.feature && n.feature !== "file_context");
    const t1 = dataNeeds[0]?.title;
    const t2 = dataNeeds[1]?.title;
    els.pointAdviceCompactText.textContent = t1 ? (t2 ? `${t1} · ${t2}` : t1) : (needs[0] && needs[0].title) || "—";
    els.pointAdviceCompact.classList.remove("hidden");
  }
}

function setReportCluster(c) {
  const v = String(c);
  if (els.reportCluster) els.reportCluster.value = v;
  renderSummary(v);
  renderZChart(v);
  renderHeuristics(v);
  renderPointLevelAdvice();
  window.dispatchEvent(new CustomEvent("equity-selection-changed"));
}

const STREET_CACHE_KEY = "useq:streetCache:v1";
let streetCache = (() => {
  try {
    return JSON.parse(localStorage.getItem(STREET_CACHE_KEY) || "{}");
  } catch {
    return {};
  }
})();

function persistStreetCache() {
  try {
    localStorage.setItem(STREET_CACHE_KEY, JSON.stringify(streetCache));
  } catch {
    // ignore storage failures
  }
}

let geocodeQueue = Promise.resolve();
function reverseGeocode(lat, lon) {
  const key = `${lat.toFixed(5)},${lon.toFixed(5)}`;
  if (streetCache[key]) return Promise.resolve(streetCache[key]);
  geocodeQueue = geocodeQueue.then(() => new Promise((res) => setTimeout(res, 1100)));
  return geocodeQueue.then(async () => {
    if (streetCache[key]) return streetCache[key];
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`;
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) throw new Error(`nominatim ${resp.status}`);
    const j = await resp.json();
    const a = j.address || {};
    const road = a.road || a.pedestrian || a.footway || a.path || "";
    const nbhd = a.neighbourhood || a.suburb || a.quarter || "";
    const desc = road ? `${road}${nbhd ? ` (${nbhd})` : ""}` : (j.display_name || "").split(",").slice(0, 2).join(", ");
    const out = { road, neighborhood: nbhd, display: desc, full: j.display_name || "" };
    streetCache[key] = out;
    persistStreetCache();
    return out;
  });
}

function streetDescriptorFor(props) {
  const lat = Number(props?.lat ?? props?.latitude);
  const lon = Number(props?.lon ?? props?.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  const key = `${lat.toFixed(5)},${lon.toFixed(5)}`;
  return streetCache[key]?.display || null;
}

function lazyAttachStreet(el, lat, lon) {
  if (!el || !Number.isFinite(lat) || !Number.isFinite(lon)) return;
  const trigger = () => {
    reverseGeocode(lat, lon)
      .then((s) => {
        if (s?.display) el.textContent = s.display;
      })
      .catch(() => {
        el.textContent = `(${lat.toFixed(4)}, ${lon.toFixed(4)})`;
      });
  };
  if (!("IntersectionObserver" in window)) {
    trigger();
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          io.unobserve(e.target);
          trigger();
        }
      }
    },
    { rootMargin: "200px" }
  );
  io.observe(el);
}

function renderCellPanel(f, ctxEl, needsEl, opts = {}) {
  if (!f || !ctxEl || !needsEl) return;
  const p = f.properties;
  const lat = Number(f.geometry?.coordinates?.[1]);
  const lon = Number(f.geometry?.coordinates?.[0]);
  const eq = Number(p.equity_score);
  const cluster = clusterName(p.cluster);
  const zRow = zRows.find((r) => String(r.cluster) === String(p.cluster));
  const cached = streetDescriptorFor({ lat, lon });
  const quad = p.lisa_quadrant || "—";
  const streetId = opts.streetId || `cellStreet_${p.grid_id ?? "x"}`;

  ctxEl.classList.remove("emptyResult");
  ctxEl.innerHTML = `
    <div class="equityCellHead">
      <div class="equityCellId">Grid #${p.grid_id ?? "?"}</div>
      <div class="equityCellBadges">
        <span class="lookupBadge lookupQuad lookupQuad--${quad}">LISA: ${quad}</span>
        <span class="lookupBadge">${cluster}</span>
        <span class="lookupBadge">Equity ${Number.isFinite(eq) ? eq.toFixed(2) : "—"}</span>
      </div>
    </div>
    <div class="equityCellGrid">
      <div class="lookupKv"><div class="lookupK">Neighborhood</div><div class="lookupV">${(p.neighborhood ?? "—") || "—"}</div></div>
      <div class="lookupKv"><div class="lookupK">Coordinates</div><div class="lookupV">${lat.toFixed(5)}°N, ${Math.abs(lon).toFixed(5)}°W</div></div>
      <div class="lookupKv lookupKv--street">
        <div class="lookupK">Street descriptor</div>
        <div class="lookupV" id="${streetId}" data-street="1">${cached || `(${lat.toFixed(4)}, ${lon.toFixed(4)})`}</div>
      </div>
      <div class="lookupKv lookupKv--street">
        <div class="lookupK">Top distinct features (vs city avg)</div>
        <div class="lookupV equityCellFeatures">${formatTop3(zRow)}</div>
      </div>
    </div>
    <div class="lookupActions">
      <a class="btn secondary" target="_blank" rel="noopener" href="https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=18/${lat}/${lon}">Open on OSM</a>
      <a class="btn secondary" target="_blank" rel="noopener" href="https://www.google.com/maps?q=${lat},${lon}">Open in Google Maps</a>
    </div>
  `;
  renderHeuristics(p.cluster, needsEl);

  const streetEl = document.getElementById(streetId);
  if (streetEl && streetEl.textContent.indexOf("(") === 0) {
    lazyAttachStreet(streetEl, lat, lon);
  }
}

function setupCellPicker(opts) {
  const nbhdSel = document.getElementById(opts.nbhdSelId);
  const cellSel = document.getElementById(opts.cellSelId);
  const ctxEl = document.getElementById(opts.ctxId);
  const needsEl = document.getElementById(opts.needsId);
  if (!nbhdSel || !cellSel || !ctxEl || !needsEl) return;

  const byNbhd = new Map();
  for (const f of geo.features) {
    if (!opts.predicate(f)) continue;
    const n = (f.properties?.neighborhood ?? "").trim() || "(unknown)";
    if (!byNbhd.has(n)) byNbhd.set(n, []);
    byNbhd.get(n).push(f);
  }
  if (!byNbhd.size) {
    ctxEl.classList.add("emptyResult");
    ctxEl.textContent = opts.emptyMessage || "No matching cells.";
    cellSel.innerHTML = "";
    nbhdSel.innerHTML = "";
    return;
  }

  const nbhds = Array.from(byNbhd.entries()).sort((a, b) => {
    const wa = a[1].reduce((s, f) => s + (f.properties.lisa_quadrant === "LL" ? 2 : 1), 0);
    const wb = b[1].reduce((s, f) => s + (f.properties.lisa_quadrant === "LL" ? 2 : 1), 0);
    return wb - wa;
  });
  nbhdSel.innerHTML = nbhds
    .map(([n, list]) => `<option value="${encodeURIComponent(n)}">${n} (${list.length})</option>`)
    .join("");

  const renderForCell = () => {
    const id = String(cellSel.value);
    const f = geo.features.find((ft) => String(ft.properties?.grid_id) === id);
    if (!f) {
      ctxEl.classList.add("emptyResult");
      ctxEl.textContent = opts.emptyMessage || "No matching cell.";
      needsEl.innerHTML = "";
      return;
    }
    renderCellPanel(f, ctxEl, needsEl, { streetId: opts.streetId });
  };

  const populateCells = () => {
    const n = decodeURIComponent(nbhdSel.value);
    const cells = (byNbhd.get(n) || []).slice().sort(opts.sortCells);
    cellSel.innerHTML = cells.map((f) => `<option value="${f.properties.grid_id}">${opts.cellLabel(f)}</option>`).join("");
    renderForCell();
  };

  nbhdSel.addEventListener("change", populateCells);
  cellSel.addEventListener("change", renderForCell);
  populateCells();
}

function setupLisaCellPicker() {
  setupCellPicker({
    nbhdSelId: "lisaNbhdSel",
    cellSelId: "lisaCellSel",
    ctxId: "lisaCellContext",
    needsId: "lisaCellNeeds",
    streetId: "lisaCellStreet",
    emptyMessage: "No statistically significant low-equity cells found.",
    predicate: (f) => {
      const q = f.properties?.lisa_quadrant;
      return q === "LL" || q === "LH";
    },
    sortCells: (a, b) => {
      const qa = a.properties.lisa_quadrant;
      const qb = b.properties.lisa_quadrant;
      if (qa !== qb) return qa === "LL" ? -1 : 1;
      return Number(a.properties.equity_score) - Number(b.properties.equity_score);
    },
    cellLabel: (f) => `${f.properties.grid_id}`,
  });
}

function setupCellLookup() {
  const input = document.getElementById("cellLookupInput");
  const btn = document.getElementById("cellLookupBtn");
  const out = document.getElementById("cellLookupResult");
  if (!input || !btn || !out) return;

  const findCell = (idStr) => {
    const id = String(idStr).trim();
    if (!id) return null;
    return geo?.features?.find((f) => String(f.properties?.grid_id) === id) || null;
  };

  const doLookup = async () => {
    out.classList.remove("emptyResult");
    const f = findCell(input.value);
    if (!f) {
      out.innerHTML = `<div class="lookupErr">No grid cell found for "<b>${input.value || "—"}</b>". Try a numeric grid_id from the map.</div>`;
      return;
    }
    const p = f.properties;
    const lat = Number(f.geometry?.coordinates?.[1]);
    const lon = Number(f.geometry?.coordinates?.[0]);
    const eq = Number(p.equity_score);
    const cluster = clusterName(p.cluster);
    const quad = p.lisa_quadrant || "—";
    out.innerHTML = `
      <div class="lookupHead">
        <div class="lookupGridId">Grid #${p.grid_id ?? "?"}</div>
        <div class="lookupBadges">
          <span class="lookupBadge">${cluster}</span>
          <span class="lookupBadge lookupQuad lookupQuad--${quad}">LISA: ${quad}</span>
          <span class="lookupBadge">Equity ${Number.isFinite(eq) ? eq.toFixed(2) : "—"}</span>
        </div>
      </div>
      <div class="lookupGrid">
        <div class="lookupKv"><div class="lookupK">Neighborhood</div><div class="lookupV">${(p.neighborhood ?? "—") || "—"}</div></div>
        <div class="lookupKv"><div class="lookupK">Coordinates</div><div class="lookupV">${lat.toFixed(5)}°N, ${Math.abs(lon).toFixed(5)}°W</div></div>
        <div class="lookupKv lookupKv--street"><div class="lookupK">Street descriptor</div><div class="lookupV" id="lookupStreet">resolving via OpenStreetMap…</div></div>
      </div>
      <div class="lookupActions">
        <a class="btn secondary" target="_blank" rel="noopener" href="https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=18/${lat}/${lon}">Open on OSM</a>
        <a class="btn secondary" target="_blank" rel="noopener" href="https://www.google.com/maps?q=${lat},${lon}">Open in Google Maps</a>
      </div>
    `;
    try {
      const s = await reverseGeocode(lat, lon);
      const el = document.getElementById("lookupStreet");
      if (el) el.textContent = s.display || s.full || "(unresolved)";
    } catch (err) {
      const el = document.getElementById("lookupStreet");
      if (el) el.textContent = `(geocoding failed: ${err.message || err})`;
    }
  };

  btn.addEventListener("click", doLookup);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doLookup();
  });
}

function setupEquityCellPicker() {
  const nbhdSel = document.getElementById("equityCellNbhd");
  const cellSel = document.getElementById("equityCellId");
  const ctxEl = document.getElementById("equityCellContext");
  const needsEl = document.getElementById("equityCellNeeds");
  if (!nbhdSel || !cellSel || !ctxEl || !needsEl) return;

  const llByNbhd = new Map();
  for (const f of geo.features) {
    if (f.properties?.lisa_quadrant !== "LL") continue;
    const n = (f.properties?.neighborhood ?? "").trim() || "(unknown)";
    if (!llByNbhd.has(n)) llByNbhd.set(n, []);
    llByNbhd.get(n).push(f);
  }
  if (!llByNbhd.size) {
    ctxEl.textContent = "No LL (statistically significant low-equity) cells found in this dataset.";
    return;
  }

  const nbhds = Array.from(llByNbhd.entries()).sort((a, b) => b[1].length - a[1].length);
  nbhdSel.innerHTML = nbhds
    .map(([n, list]) => `<option value="${encodeURIComponent(n)}">${n} (${list.length} LL ${list.length === 1 ? "cell" : "cells"})</option>`)
    .join("");

  const renderForCell = () => {
    const id = String(cellSel.value);
    const f = geo.features.find((ft) => String(ft.properties?.grid_id) === id);
    if (!f) {
      ctxEl.classList.add("emptyResult");
      ctxEl.textContent = "No matching LL cell.";
      needsEl.innerHTML = "";
      return;
    }
    renderCellPanel(f, ctxEl, needsEl, { streetId: "equityCellStreet" });
  };

  const populateCells = () => {
    const n = decodeURIComponent(nbhdSel.value);
    const cells = (llByNbhd.get(n) || []).slice().sort((a, b) => Number(a.properties.equity_score) - Number(b.properties.equity_score));
    cellSel.innerHTML = cells.map((f) => `<option value="${f.properties.grid_id}">${f.properties.grid_id}</option>`).join("");
    renderForCell();
  };

  nbhdSel.addEventListener("change", populateCells);
  cellSel.addEventListener("change", renderForCell);
  populateCells();
}

function renderLowEquityCongregations() {
  const target = document.getElementById("lowEquityCongregations");
  if (!target) return;
  if (!geo?.features?.length) {
    target.textContent = "—";
    return;
  }

  const byNbhd = new Map();
  for (const f of geo.features) {
    const q = f.properties?.lisa_quadrant;
    if (q !== "LL" && q !== "LH") continue;
    const name = (f.properties?.neighborhood ?? "").trim() || "(unknown)";
    if (!byNbhd.has(name)) byNbhd.set(name, { name, ll: 0, lh: 0, scores: [] });
    const e = byNbhd.get(name);
    if (q === "LL") e.ll += 1;
    else e.lh += 1;
    const eq = Number(f.properties?.equity_score);
    if (Number.isFinite(eq)) e.scores.push(eq);
  }

  const ranked = Array.from(byNbhd.values())
    .map((e) => ({
      ...e,
      total: e.ll + e.lh,
      meanScore: e.scores.length ? e.scores.reduce((a, b) => a + b, 0) / e.scores.length : NaN,
    }))
    .filter((e) => e.total > 0)
    .sort((a, b) => b.ll * 2 + b.lh - (a.ll * 2 + a.lh))
    .slice(0, 10);

  if (!ranked.length) {
    target.textContent = "No statistically significant low-equity cells found.";
    return;
  }
  const maxTotal = Math.max(...ranked.map((r) => r.total));

  target.innerHTML = `
    <div class="congHead">
      <div class="congCol congCol--rank">#</div>
      <div class="congCol congCol--name">Neighborhood</div>
      <div class="congCol congCol--bar">Underserved cells</div>
      <div class="congCol congCol--n">LL</div>
      <div class="congCol congCol--n">LH</div>
      <div class="congCol congCol--score">Mean eq.</div>
    </div>
    ${ranked
      .map((r, i) => {
        const llW = (r.ll / maxTotal) * 100;
        const lhW = (r.lh / maxTotal) * 100;
        return `
          <button type="button" class="congRow" data-nbhd="${encodeURIComponent(r.name)}">
            <div class="congCol congCol--rank">${i + 1}</div>
            <div class="congCol congCol--name">${r.name}</div>
            <div class="congCol congCol--bar">
              <div class="congBarTrack">
                <div class="congBarLL" style="width:${llW}%" title="LL: ${r.ll}"></div>
                <div class="congBarLH" style="width:${lhW}%" title="LH: ${r.lh}"></div>
              </div>
            </div>
            <div class="congCol congCol--n congCol--n-ll">${r.ll}</div>
            <div class="congCol congCol--n congCol--n-lh">${r.lh}</div>
            <div class="congCol congCol--score">${Number.isFinite(r.meanScore) ? r.meanScore.toFixed(2) : "—"}</div>
          </button>`;
      })
      .join("")}
    <div class="congLegend">
      <span class="congSwatch congSwatch--ll"></span> LL = significant low-equity cluster
      &nbsp;&nbsp;
      <span class="congSwatch congSwatch--lh"></span> LH = low-equity pocket in well-served surroundings
    </div>
  `;

  target.querySelectorAll(".congRow[data-nbhd]").forEach((row) => {
    row.addEventListener("click", () => {
      const name = decodeURIComponent(row.dataset.nbhd);
      target.querySelectorAll(".congRow").forEach((r) => r.classList.toggle("isActive", r === row));
      renderLowEquityCellsForNeighborhood(name);
    });
  });
}

function renderLowEquityCellsForNeighborhood(name) {
  const out = document.getElementById("lowEquityCells");
  if (!out) return;
  const cells = geo.features.filter((f) => {
    const q = f.properties?.lisa_quadrant;
    if (q !== "LL" && q !== "LH") return false;
    const n = (f.properties?.neighborhood ?? "").trim() || "(unknown)";
    return n === name;
  });
  if (!cells.length) {
    out.innerHTML = `<div class="emptyText">No underserved cells found in ${name}.</div>`;
    return;
  }

  const ll = cells.filter((f) => f.properties.lisa_quadrant === "LL").sort((a, b) => Number(a.properties.equity_score) - Number(b.properties.equity_score));
  const lh = cells.filter((f) => f.properties.lisa_quadrant === "LH").sort((a, b) => Number(a.properties.equity_score) - Number(b.properties.equity_score));
  const sample = ll.concat(lh).slice(0, 6);

  out.innerHTML = `
    <div class="lowEqTitle">Sample of underserved cells in <b>${name}</b>
      <span class="lowEqCount">(${cells.length} total &middot; showing ${sample.length})</span>
    </div>
    <ul class="lowEqList">
      ${sample
        .map((f) => {
          const p = f.properties;
          const lat = Number(f.geometry?.coordinates?.[1]);
          const lon = Number(f.geometry?.coordinates?.[0]);
          const eq = Number(p.equity_score);
          const cached = streetDescriptorFor({ lat, lon });
          return `
            <li class="lowEqCell" data-lat="${lat}" data-lon="${lon}">
              <span class="lowEqQuad lowEqQuad--${p.lisa_quadrant}">${p.lisa_quadrant}</span>
              <span class="lowEqId">#${p.grid_id ?? "?"}</span>
              <span class="lowEqEq">eq ${Number.isFinite(eq) ? eq.toFixed(2) : "—"}</span>
              <span class="lowEqStreet" data-street="1">${cached || `(${lat.toFixed(4)}, ${lon.toFixed(4)})`}</span>
            </li>`;
        })
        .join("")}
    </ul>
  `;

  out.querySelectorAll(".lowEqCell").forEach((li) => {
    const street = li.querySelector(".lowEqStreet[data-street='1']");
    if (!street || street.textContent.indexOf("(") !== 0) return;
    lazyAttachStreet(street, Number(li.dataset.lat), Number(li.dataset.lon));
  });
}

function ensureNeighborhoodOverlay() {
  if (!map || !nbhdGeo || nbhdLayer) return;
  nbhdLayer = L.geoJSON(nbhdGeo, {
    interactive: false,
    style: () => ({
      color: "rgba(255,255,255,.55)",
      weight: 1.2,
      opacity: 0.85,
      fill: true,
      fillColor: "#ffffff",
      fillOpacity: 0.0,
      dashArray: "3,3",
    }),
  }).addTo(map);
  if (nbhdLayer.bringToBack) nbhdLayer.bringToBack();
}

function rebuildLayer() {
  if (!map || !geo) return;
  if (layer) layer.remove();
  layer = L.geoJSON(geo, {
    filter: (feature) => passesFilters(feature.properties ?? {}),
    pointToLayer: (feature, latlng) => {
      const props = feature.properties ?? {};
      const style = markerStyle(props);
      return L.circleMarker(latlng, {
        radius: 5,
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.85,
        ...style,
      });
    },
    onEachFeature: (feature, l) => {
      const p = feature.properties ?? {};
      const pct = rawToPercent(Number(p.equity_score));
      // Keep page scroll fixed while clicking map points.
      l.on("click", () => {
        const y = window.scrollY;
        const x = window.scrollX;
        setSelection(p);
        requestAnimationFrame(() => window.scrollTo(x, y));
      });
      l.bindTooltip(
        `<div style="font-family:ui-sans-serif,system-ui;font-size:12px">
          <div><b>${p.grid_id ?? "grid"}</b></div>
          <div>${clusterName(p.cluster)}</div>
          <div>Equity: ${pct !== null ? `${pct}th pctile` : "—"}</div>
        </div>`,
        { sticky: true }
      );
    },
  }).addTo(map);
}

async function init() {
  const [m, g, summary, z, paJ, nbhd] = await Promise.all([
    fetchJson(DATA_META),
    fetchJson(DATA_GEOJSON),
    parseCsv(DATA_SUMMARY),
    parseCsv(DATA_Z),
    fetchJson(DATA_POINT_ADVICE).catch(() => null),
    fetchJson(DATA_NBHD).catch(() => null),
  ]);
  pointAdviceByGrid = paJ?.by_grid != null ? paJ.by_grid : null;
  meta = m;
  geo = g;
  summaryRows = summary;
  zRows = z;
  nbhdGeo = nbhd;
  sortedScores = geo.features
    .map((f) => Number(f.properties?.equity_score))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  window.dispatchEvent(
    new CustomEvent("equity-dashboard-ready", {
      detail: { meta, geo, summaryRows },
    })
  );

  map = L.map("map", { zoomControl: true, minZoom: 12 }).setView([37.77, -122.44], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  clearSelection();
  renderLegend();
  renderPca();
  setReportCluster("0");
  setupLisaCellPicker();
  renderLowEquityCongregations();
  setupEquityCellPicker();
  setupCellLookup();
  ensureNeighborhoodOverlay();
  rebuildLayer();

  const syncHeroScrollCue = () => {
    document.body.classList.toggle("heroCollapsed", window.scrollY > 320);
  };
  window.addEventListener("scroll", syncHeroScrollCue, { passive: true });
  syncHeroScrollCue();
}

els.applyFilters?.addEventListener("click", () => {
  renderLegend();
  rebuildLayer();
});
els.colorMode?.addEventListener("change", () => {
  renderLegend();
  rebuildLayer();
  if (selectedPointProps) setSelection(selectedPointProps);
});
els.clearSelection?.addEventListener("click", () => clearSelection());
els.reportCluster?.addEventListener("change", (e) => setReportCluster(e.target.value));
els.clusterLink?.addEventListener("click", (e) => {
  e.preventDefault();
  const target = document.querySelector(els.clusterLink.getAttribute("href"));
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
});

init().catch((err) => {
  console.error(err);
  alert(
    `Failed to load dashboard data.\n\n${String(err?.message || err)}\n\n` +
      "If you opened this as a file, run: python -m http.server 5173 --directory docs\n" +
      "Ensure docs/outputs contains the pipeline files (run run_pipeline.py --output-dir docs/outputs)."
  );
});

