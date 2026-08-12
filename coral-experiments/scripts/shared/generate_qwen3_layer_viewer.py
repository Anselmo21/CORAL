"""Generate a self-contained HTML viewer showing answer distributions per layer
per sampled sublattice, for all 4 Qwen3 datasets.

For each case in results.json, computes — for every layer of the sublattice
(L0=empty set … L{size}=start_node) — how many nodes at that layer have each
plausible answer as top-1. The CF layer (from BFS) is highlighted.

Output: artifacts/qwen3_layer_viewer.html
Usage:  conda run -n coral python3 -m scripts.shared.generate_qwen3_layer_viewer
"""
from __future__ import annotations

import json
from itertools import combinations

from search.lattice import combo_key

DATASETS = [
    ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",   "MSMARCO Criteria"),
    ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",    "MSMARCO Pragma"),
    ("curated_criteria",  "data/qwen3-32b/curated-criteria-qwen3-32b.json",    "Curated Criteria"),
    ("curated_pragma",    "data/qwen3-32b/curated-pragmas-qwen3-32b.json",     "Curated Pragma"),
]

OUT_PATH = "artifacts/qwen3_layer_viewer.html"


def build_cases(ds_key: str, data_path: str) -> list[dict]:
    results_path = f"logged_results/qwen3_{ds_key}/results.json"
    results = json.load(open(results_path))["results"]
    raw_data = json.load(open(data_path))

    cases = []
    for r in results:
        if not r["bfs"]["found"]:
            continue

        qid        = r["qid"]
        start_node = r["start_node"]
        size       = r["size"]
        bfs_layer  = r["bfs"]["layer"]  # item count of CF node

        if qid not in raw_data:
            continue
        v       = raw_data[qid]
        rbc     = v["raw_answers"]
        answers = list(v["plausible_answers"])
        a2i     = {a: i for i, a in enumerate(answers)}

        # For each layer k (0..size), count how many sublattice nodes have each answer as top-1
        layer_counts = [[0] * len(answers) for _ in range(size + 1)]
        other_counts = [0] * (size + 1)

        for k in range(size + 1):
            for subset in combinations(start_node, k):
                key = combo_key(list(subset))
                ranking = rbc.get(key)
                if not ranking:
                    continue
                top1 = ranking[0]
                idx = a2i.get(top1)
                if idx is not None:
                    layer_counts[k][idx] += 1
                else:
                    other_counts[k] += 1

        cases.append({
            "case_id":    r["case_id"],
            "qid":        qid,
            "question":   v.get("question", ""),
            "size":       size,
            "start_node": start_node,
            "answers":    answers,
            "bfs_layer":  bfs_layer,
            "bfs_key":    r["bfs"]["key"],
            "layers":     layer_counts,
            "other":      other_counts,
        })

    return cases


def build_data() -> dict:
    out = {}
    for ds_key, data_path, label in DATASETS:
        print(f"  {label}…")
        out[ds_key] = {"label": label, "cases": build_cases(ds_key, data_path)}
    return out


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qwen3 Layer Distribution Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f0f2f5;color:#333;height:100vh;display:flex;flex-direction:column;overflow:hidden}

.tab-bar{display:flex;background:#16213e;flex-shrink:0;padding:0 8px}
.tab{padding:12px 20px;cursor:pointer;color:#8a9bb0;font-size:13px;font-weight:500;
     border-bottom:3px solid transparent;transition:all .15s;user-select:none}
.tab:hover{color:#d0dce8}
.tab.active{color:#fff;border-bottom-color:#4e79a7}

.layout{display:flex;flex:1;overflow:hidden}

.sidebar{width:280px;background:#fff;border-right:1px solid #e4e8ee;
         display:flex;flex-direction:column;flex-shrink:0}
.sidebar-hdr{padding:10px 14px;background:#f7f8fa;border-bottom:1px solid #e4e8ee;
             font-size:11px;font-weight:700;color:#888;letter-spacing:.6px;text-transform:uppercase;
             display:flex;justify-content:space-between;align-items:center}
.size-filter{display:flex;gap:4px;flex-wrap:wrap;padding:8px 14px;border-bottom:1px solid #eee}
.size-btn{padding:2px 8px;font-size:11px;border:1px solid #d0d5dd;border-radius:4px;
          background:#fff;cursor:pointer;color:#555;transition:all .12s}
.size-btn.active{background:#4e79a7;color:#fff;border-color:#4e79a7}
.case-list{overflow-y:auto;flex:1}
.citem{padding:9px 14px;cursor:pointer;border-bottom:1px solid #f0f2f5;transition:background .1s}
.citem:hover{background:#f0f4ff}
.citem.active{background:#e8f0fe;border-left:3px solid #4e79a7;padding-left:11px}
.citem-meta{font-size:10px;color:#9aa5b4;letter-spacing:.3px;margin-bottom:2px}
.citem-q{font-size:12px;color:#444;line-height:1.35;
         display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

.content{flex:1;overflow-y:auto;padding:24px 32px 40px}
.placeholder{color:#bbb;text-align:center;margin-top:80px;font-size:14px}

.q-text{font-size:16px;font-weight:600;color:#16213e;line-height:1.45;margin-bottom:6px}
.meta-row{font-size:12px;color:#888;margin-bottom:16px}

.card{background:#fff;border-radius:8px;padding:18px 22px;
      box-shadow:0 1px 4px rgba(0,0,0,.07);margin-bottom:18px}
.card-title{font-size:11px;font-weight:700;color:#9aa5b4;letter-spacing:.7px;
            text-transform:uppercase;margin-bottom:12px}

.start-node{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:4px}
.node-chip{padding:3px 10px;border-radius:12px;font-size:12px;background:#eef2fb;
           color:#3457a0;border:1px solid #c9d6f0}

.legend{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.leg-item{display:flex;align-items:center;gap:5px;font-size:12px;color:#444}
.leg-swatch{width:12px;height:12px;border-radius:3px;flex-shrink:0}

/* bar chart */
.chart-area{overflow-x:auto}
.chart-svg-wrap{min-width:100%}

.cf-label{font-size:11px;font-weight:600;color:#c0392b;margin-top:10px}
</style>
</head>
<body>

<div class="tab-bar" id="tabs"></div>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-hdr">
      <span>Cases</span>
      <span id="case-count" style="font-weight:400;font-size:10px"></span>
    </div>
    <div class="size-filter" id="size-filter"></div>
    <div class="case-list" id="case-list"></div>
  </div>
  <div class="content" id="content">
    <div class="placeholder">Select a dataset and case</div>
  </div>
</div>

<script>
const DATA = __DATA__;

const COLORS = [
  "#4e79a7","#f28e2b","#59a14f","#e15759","#76b7b2",
  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"
];
const OTHER_COLOR = "#cccccc";

let activeDs = null;
let activeSize = null;
let activeCase = null;

function init() {
  const dsKeys = Object.keys(DATA);

  // Tabs
  const tabBar = document.getElementById("tabs");
  dsKeys.forEach((k, i) => {
    const t = document.createElement("div");
    t.className = "tab" + (i === 0 ? " active" : "");
    t.textContent = DATA[k].label;
    t.onclick = () => selectDs(k);
    t.dataset.key = k;
    tabBar.appendChild(t);
  });

  selectDs(dsKeys[0]);
}

function selectDs(key) {
  activeDs = key;
  activeSize = null;
  activeCase = null;

  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.key === key));

  renderSizeFilter();
  renderCaseList();
  document.getElementById("content").innerHTML =
    '<div class="placeholder">Select a case</div>';
}

function renderSizeFilter() {
  const cases = DATA[activeDs].cases;
  const sizes = [...new Set(cases.map(c => c.size))].sort((a,b) => a-b);
  const el = document.getElementById("size-filter");
  el.innerHTML = "";

  const allBtn = document.createElement("button");
  allBtn.className = "size-btn" + (activeSize === null ? " active" : "");
  allBtn.textContent = "All";
  allBtn.onclick = () => { activeSize = null; renderSizeFilter(); renderCaseList(); };
  el.appendChild(allBtn);

  sizes.forEach(s => {
    const btn = document.createElement("button");
    btn.className = "size-btn" + (activeSize === s ? " active" : "");
    btn.textContent = `n=${s}`;
    btn.onclick = () => { activeSize = s; renderSizeFilter(); renderCaseList(); };
    el.appendChild(btn);
  });
}

function renderCaseList() {
  const cases = DATA[activeDs].cases.filter(
    c => activeSize === null || c.size === activeSize);
  const el = document.getElementById("case-list");
  el.innerHTML = "";
  document.getElementById("case-count").textContent = `${cases.length} cases`;

  cases.forEach(c => {
    const div = document.createElement("div");
    div.className = "citem" + (activeCase === c.case_id ? " active" : "");
    div.innerHTML = `<div class="citem-meta">QID ${c.qid} &nbsp;·&nbsp; n=${c.size} &nbsp;·&nbsp; CF=L${c.bfs_layer}</div>
                     <div class="citem-q">${c.question}</div>`;
    div.onclick = () => selectCase(c.case_id);
    el.appendChild(div);
  });
}

function selectCase(caseId) {
  activeCase = caseId;
  renderCaseList();
  const c = DATA[activeDs].cases.find(x => x.case_id === caseId);
  renderCase(c);
}

function renderCase(c) {
  const nAnswers = c.answers.length;
  const hasOther = c.other.some(x => x > 0);
  const allAnswers = hasOther ? [...c.answers, "Other"] : c.answers;
  const allColors  = hasOther ? [...COLORS.slice(0, nAnswers), OTHER_COLOR]
                              : COLORS.slice(0, nAnswers);

  // Legend
  const legendHtml = allAnswers.map((a, i) =>
    `<div class="leg-item"><div class="leg-swatch" style="background:${allColors[i]}"></div>${a}</div>`
  ).join("");

  // Start node chips
  const nodeHtml = c.start_node.map(s =>
    `<span class="node-chip">${s}</span>`).join("");

  // CF node items
  const cfItems = c.bfs_key ? c.bfs_key.split(", ") : [];
  const cfHtml  = cfItems.length
    ? cfItems.map(s => `<span class="node-chip" style="background:#fdecea;color:#c0392b;border-color:#f5c6c3">${s}</span>`).join("")
    : '<span style="font-size:12px;color:#c0392b">∅ (empty set)</span>';

  const svgHtml = renderBarChart(c, allColors, hasOther);

  document.getElementById("content").innerHTML = `
    <div class="q-text">${c.question}</div>
    <div class="meta-row">QID ${c.qid} &nbsp;·&nbsp; Sublattice size ${c.size} &nbsp;·&nbsp; CF at L${c.bfs_layer}</div>

    <div class="card">
      <div class="card-title">Start Node (L${c.size})</div>
      <div class="start-node">${nodeHtml}</div>
    </div>

    <div class="card">
      <div class="card-title">CF Node (L${c.bfs_layer})</div>
      <div class="start-node">${cfHtml}</div>
    </div>

    <div class="card">
      <div class="card-title">Answer Distribution per Layer</div>
      <div class="legend">${legendHtml}</div>
      <div class="chart-area">${svgHtml}</div>
    </div>
  `;
}

function renderBarChart(c, allColors, hasOther) {
  const size = c.size;
  const nLayers = size + 1;
  const nAnswers = c.answers.length;

  // Build counts per layer: list of {answer_idx -> count}
  // Max total per layer = C(size, k)
  const maxCount = Math.max(...c.layers.map((row, k) => {
    const total = row.reduce((a,b)=>a+b,0) + c.other[k];
    return total;
  }));

  const BAR_W  = 48;
  const GAP    = 14;
  const HEIGHT = 200;
  const LABEL_H = 24;
  const PAD_L  = 36;
  const PAD_R  = 16;
  const PAD_TOP = 10;

  const totalW = PAD_L + nLayers * (BAR_W + GAP) - GAP + PAD_R;
  const totalH = PAD_TOP + HEIGHT + LABEL_H + 24;

  let rects = "";
  let labels = "";
  let cfLine = "";

  for (let k = 0; k <= size; k++) {
    const x = PAD_L + k * (BAR_W + GAP);
    const counts = [...c.layers[k], ...(hasOther ? [c.other[k]] : [])];
    const total  = counts.reduce((a,b)=>a+b,0);
    let y = PAD_TOP + HEIGHT;

    counts.forEach((cnt, i) => {
      if (cnt === 0) return;
      const h = (cnt / maxCount) * HEIGHT;
      y -= h;
      rects += `<rect x="${x}" y="${y.toFixed(1)}" width="${BAR_W}" height="${h.toFixed(1)}"
                      fill="${allColors[i]}" opacity="${k === c.bfs_layer ? 1 : 0.65}"/>`;
    });

    // Layer label
    const isCF = k === c.bfs_layer;
    labels += `<text x="${x + BAR_W/2}" y="${PAD_TOP + HEIGHT + 16}"
                     text-anchor="middle" font-size="11"
                     font-weight="${isCF ? 700 : 400}"
                     fill="${isCF ? '#c0392b' : '#666'}">L${k}</text>`;

    // CF marker
    if (isCF) {
      cfLine += `<line x1="${x-4}" x2="${x + BAR_W+4}"
                       y1="${PAD_TOP - 4}" y2="${PAD_TOP - 4}"
                       stroke="#c0392b" stroke-width="2" stroke-dasharray="4,2"/>
                 <text x="${x + BAR_W/2}" y="${PAD_TOP - 8}"
                       text-anchor="middle" font-size="10" fill="#c0392b" font-weight="700">CF</text>`;
    }
  }

  // Y-axis label
  const yAxis = `<text x="14" y="${PAD_TOP + HEIGHT/2}" text-anchor="middle"
                        font-size="10" fill="#999" transform="rotate(-90,14,${PAD_TOP + HEIGHT/2})">nodes</text>
                 <text x="${PAD_L - 4}" y="${PAD_TOP}" text-anchor="end" font-size="10" fill="#aaa">${maxCount}</text>
                 <text x="${PAD_L - 4}" y="${PAD_TOP + HEIGHT}" text-anchor="end" font-size="10" fill="#aaa">0</text>`;

  return `<svg width="${totalW}" height="${totalH}" style="display:block">
    ${yAxis}${rects}${labels}${cfLine}
  </svg>`;
}

function filteredCases() {
  return DATA[activeDs].cases.filter(
    c => activeSize === null || c.size === activeSize);
}

document.addEventListener("keydown", e => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  const cases = filteredCases();
  if (!cases.length) return;
  const idx = activeCase === null ? -1 : cases.findIndex(c => c.case_id === activeCase);
  let next;
  if (e.key === "ArrowLeft")  next = idx <= 0 ? cases.length - 1 : idx - 1;
  if (e.key === "ArrowRight") next = idx >= cases.length - 1 ? 0 : idx + 1;
  selectCase(cases[next].case_id);
  const items = document.querySelectorAll(".citem");
  if (items[next]) items[next].scrollIntoView({block:"nearest"});
});

init();
</script>
</body>
</html>
"""


def main() -> None:
    print("Building data…")
    data = build_data()
    total = sum(len(v["cases"]) for v in data.values())
    print(f"  {total} cases total")

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Written → {OUT_PATH}")


if __name__ == "__main__":
    main()
