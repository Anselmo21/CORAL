"""Generate a self-contained HTML viewer for Qwen3 lattice distributions + ERA entropy drill-down.

For each of the 4 datasets, for every size-10 lattice that passes the >=2 removal filter,
computes layer-by-layer answer distribution AND ERA (softmax, budget=4, max_jump) telemetry.

Click a layer bar in the distribution chart to see Phase 1 entropy scores per node,
the ideal max-jump cutoff line, and which nodes were evaluated in Phase 2.

Output: artifacts/qwen3_hasse_viewer_era.html
Usage:  conda run -n coral python3 -m scripts.shared.generate_qwen3_viewer_era
"""
from __future__ import annotations

import json
import os
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key, iter_search_layers
from search.rank import _get_child_voters, aggregator_default, _store_estimate
from search.heuristics import softmax_entropy, max_jump_cutoff

FILES = [
    ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "MSMARCO Criteria"),
    ("curated_criteria",  "data/qwen3-32b/curated-criteria-qwen3-32b.json",   "Curated Criteria"),
    ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",   "MSMARCO Pragma"),
    ("curated_pragma",    "data/qwen3-32b/curated-pragmas-qwen3-32b.json",    "Curated Pragma"),
]

MIN_REMOVED = 2
ERA_BUDGET  = 4
OUT_PATH    = "artifacts/qwen3_hasse_viewer_era.html"


# ---------------------------------------------------------------------------
# Layer map helper
# ---------------------------------------------------------------------------

def _key_to_layer_map(criteria: list[str]) -> dict[str, int]:
    m: dict[str, int] = {"": 0}
    for k in range(1, len(criteria) + 1):
        for combo in combinations(criteria, k):
            m[combo_key(list(combo))] = k
    return m


# ---------------------------------------------------------------------------
# ERA telemetry
# ---------------------------------------------------------------------------

def collect_era_telemetry(
    criteria: list[str],
    rbc: dict[str, list[str]],
    bfs_cf_key: str | None,
    display_answers: list[str],
    budget: int = ERA_BUDGET,
) -> dict[int, dict]:
    """Replay ERA (softmax, max_jump, budget) and return per-layer telemetry.

    Returns dict: layer_size (int) → {
        "nodes": [[bitmask, entropy_4dp, flags, ans_idx], ...],  # sorted entropy desc
        "cut":   ideal_cutoff_index                               # max_jump ignoring budget
    }

    flags is a 4-bit int:
      bit 0 (1): Phase 1 selected (budget was spent here)
      bit 1 (2): Phase 2 evaluated (LLM call in search)
      bit 2 (4): ERA counterfactual
      bit 3 (8): BFS counterfactual
    actual_ans_idx: index into display_answers of the ground-truth top-1 from rbc (-1 if missing)
    """
    s = len(criteria)
    start_node = list(criteria)
    start_key = combo_key(start_node)

    if start_key not in rbc:
        return {}

    baseline_ranking = rbc[start_key]
    if not baseline_ranking:
        return {}
    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)

    # Pre-load singletons + empty set
    known: dict[str, list[str]] = {}
    estimated_scores: dict[str, dict[str, float]] = {}
    entropies_map: dict[str, float] = {}
    a2i = {a: i for i, a in enumerate(display_answers)}

    for c in criteria:
        if c in rbc:
            known[c] = rbc[c]
    if "" in rbc:
        known[""] = rbc[""]

    # ── Phase 1: bottom-up entropy estimation + budget-capped lookups ────────
    budget_remaining = budget
    phase1_selected: set[str] = set()
    layer_data: dict[int, list[tuple[str, float]]] = {}

    for layer_size in range(2, s):
        layer_nodes: list[tuple[str, float]] = []
        for combo in combinations(range(s), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)
            voter_scores = _get_child_voters(subset, known, estimated_scores, answers)
            if not voter_scores:
                continue
            ranking, scores = aggregator_default(voter_scores, answers)
            if not scores:
                continue
            estimated_scores[key] = _store_estimate(ranking, scores, answers, True)
            ent = softmax_entropy(scores, num_voters=len(voter_scores))
            entropies_map[key] = ent
            layer_nodes.append((key, ent))

        layer_data[layer_size] = layer_nodes

        for sel_key in max_jump_cutoff(layer_nodes, budget_remaining):
            if sel_key in rbc:
                known[sel_key] = rbc[sel_key]
                budget_remaining -= 1
                phase1_selected.add(sel_key)

    # ── Phase 2: top-down search with pruning ────────────────────────────────
    phase2_evaluated: set[str] = set()
    era_cf_key: str | None = None
    outer_done = False

    for _layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        if outer_done:
            break
        unknown_nodes: list[tuple[list[str], str, float, float]] = []

        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                if known[key] and known[key][0] != baseline_answer:
                    era_cf_key = key
                    outer_done = True
                    break
                continue
            if key not in estimated_scores:
                continue
            scores = estimated_scores[key]
            ranking = sorted(scores, key=lambda a: (-scores[a], a))
            ent = entropies_map.get(key, 0.0)
            disp = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            unknown_nodes.append((subset, key, disp, ent))

        if outer_done or not unknown_nodes:
            continue

        entropy_inputs = [(key, ent) for _, key, _, ent in unknown_nodes]
        high_entropy_keys = set(max_jump_cutoff(entropy_inputs, len(unknown_nodes)))
        candidates = [
            (subset, key, disp, ent)
            for subset, key, disp, ent in unknown_nodes
            if not (disp < 1 and key not in high_entropy_keys)
        ]
        candidates.sort(key=lambda x: (-x[2], x[3]))

        for subset, key, _, _ in candidates:
            phase2_evaluated.add(key)
            ground_truth = rbc.get(key)
            if ground_truth:
                known[key] = ground_truth
            if ground_truth and ground_truth[0] != baseline_answer:
                era_cf_key = key
                outer_done = True
                break

    # ── Build output ─────────────────────────────────────────────────────────
    bfs_cf_set = set(bfs_cf_key.split(", ")) if bfs_cf_key else set()

    result: dict[int, dict] = {}
    for layer_size, layer_nodes in layer_data.items():
        sorted_nodes = sorted(layer_nodes, key=lambda x: -x[1])
        n = len(sorted_nodes)

        # Ideal max_jump cutoff index (ignoring budget)
        if n > 1:
            ents = [e for _, e in sorted_nodes]
            gaps = [ents[i] - ents[i + 1] for i in range(n - 1)]
            ideal_cut = max(range(len(gaps)), key=lambda i: (gaps[i], i))
        else:
            ideal_cut = 0

        nodes_out = []
        for key, ent in sorted_nodes:
            subset_set = set(key.split(", ")) if key else set()
            bitmask = sum(1 << i for i, c in enumerate(criteria) if c in subset_set)
            flags = (
                (1 if key in phase1_selected   else 0) |
                (2 if key in phase2_evaluated  else 0) |
                (4 if key == era_cf_key        else 0) |
                (8 if subset_set == bfs_cf_set else 0)
            )
            actual_top1 = (rbc.get(key) or [None])[0]
            actual_ans_idx = a2i.get(actual_top1, -1)
            nodes_out.append([bitmask, round(ent, 4), flags, actual_ans_idx])

        result[layer_size] = {"nodes": nodes_out, "cut": ideal_cut}

    return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(path: str) -> list[dict]:
    raw = json.load(open(path))
    lattices = []
    for qid, v in raw.items():
        if qid == "_meta" or v.get("halted"):
            continue
        criteria = list(v["criteria"])
        if len(criteria) != 10:
            continue
        rbc = v["raw_answers"]
        if combo_key(criteria) not in rbc:
            continue

        res = bfs_search(criteria, rbc, start_node=criteria, direction="down")
        if not res["found"] or (len(criteria) - res["size"]) < MIN_REMOVED:
            continue

        answers = list(v["plausible_answers"])
        a2i     = {a: i for i, a in enumerate(answers)}
        k2l     = _key_to_layer_map(criteria)

        layer_counts = [[0] * len(answers) for _ in range(11)]
        for rk, ranking in rbc.items():
            layer = k2l.get(rk)
            if layer is None:
                continue
            top1 = ranking[0] if ranking else None
            idx  = a2i.get(top1)
            if idx is not None:
                layer_counts[layer][idx] += 1

        cf_set  = set(res["key"].split(", ")) if res["key"] else set()
        removed = [c for c in criteria if c not in cf_set]

        era = collect_era_telemetry(criteria, rbc, res["key"], answers)

        lattices.append({
            "qid":         qid,
            "question":    v.get("question", ""),
            "criteria":    criteria,
            "answers":     answers,
            "bfs_layer":   res["size"],
            "bfs_removed": removed,
            "layers":      layer_counts,
            "era":         era,
        })

    lattices.sort(key=lambda x: x["qid"])
    return lattices


def build_all_data() -> dict:
    all_data: dict = {}
    for key, path, label in FILES:
        print(f"[{label}] loading …", flush=True)
        lats = build_dataset(path)
        all_data[key] = {"label": label, "lattices": lats}
        print(f"  {len(lats)} passing lattices")
    return all_data


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qwen3-32B Lattice Viewer + ERA</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f0f2f5;color:#333;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Tabs ── */
.tab-bar{display:flex;background:#16213e;flex-shrink:0;padding:0 8px}
.tab{padding:12px 20px;cursor:pointer;color:#8a9bb0;font-size:13px;font-weight:500;
     border-bottom:3px solid transparent;transition:all .15s;user-select:none}
.tab:hover{color:#d0dce8}
.tab.active{color:#fff;border-bottom-color:#4e79a7}

/* ── Layout ── */
.layout{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
.sidebar{width:270px;background:#fff;border-right:1px solid #e4e8ee;
         display:flex;flex-direction:column;flex-shrink:0}
.sidebar-hdr{padding:12px 16px;background:#f7f8fa;border-bottom:1px solid #e4e8ee;
             font-size:11px;font-weight:700;color:#888;letter-spacing:.6px;text-transform:uppercase}
.query-list{overflow-y:auto;flex:1}
.qitem{padding:10px 16px;cursor:pointer;border-bottom:1px solid #f0f2f5;transition:background .12s}
.qitem:hover{background:#f0f4ff}
.qitem.active{background:#e8f0fe;border-left:3px solid #4e79a7;padding-left:13px}
.qitem-id{font-size:10px;font-weight:700;color:#9aa5b4;letter-spacing:.4px}
.qitem-q{font-size:12px;color:#444;margin-top:3px;line-height:1.35;
          display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

/* ── Content ── */
.content{flex:1;overflow-y:auto;padding:28px 36px 40px}
.placeholder{color:#bbb;text-align:center;margin-top:80px;font-size:14px}
.q-text{font-size:17px;font-weight:600;color:#16213e;line-height:1.45;margin-bottom:20px}

/* ── Section label ── */
.sec{font-size:10px;font-weight:700;color:#9aa5b4;letter-spacing:.8px;
     text-transform:uppercase;margin-bottom:8px}

/* ── Answers ── */
.answers{display:flex;flex-direction:column;gap:7px;margin-bottom:20px}
.achip{display:flex;align-items:flex-start;gap:10px;padding:9px 13px;
       background:#f7f8fa;border-radius:6px;font-size:13px;line-height:1.45;border:1px solid #eaecef}
.aswatch{width:13px;height:13px;border-radius:3px;flex-shrink:0;margin-top:2px}

/* ── BFS banner ── */
.bfs-banner{background:#fffbe6;border:1px solid #ffe58f;border-radius:6px;
            padding:10px 14px;margin-bottom:22px;font-size:13px;line-height:1.5}
.bfs-banner strong{color:#c67d00}
.rtag{display:inline-block;background:#fff3cc;border:1px solid #ffe58f;
      border-radius:4px;padding:2px 8px;margin:2px 3px;font-size:11px;color:#614700}

/* ── Chart card ── */
.chart-card{background:#fff;border-radius:8px;padding:20px 24px;
            box-shadow:0 1px 4px rgba(0,0,0,.07);margin-bottom:20px}
.chart-title{font-size:13px;font-weight:600;color:#555;margin-bottom:4px}
.chart-sub{font-size:11px;color:#aaa;margin-bottom:14px}
.chart-wrap{overflow-x:auto}

/* ── ERA panel ── */
#era-panel{margin-bottom:20px}

/* ── Legend ── */
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:12px}
.leg-item{display:flex;align-items:center;gap:6px;font-size:12px;color:#555;max-width:340px}
.leg-dot{width:12px;height:12px;border-radius:3px;flex-shrink:0}

/* ── Criteria tags ── */
.ctags{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.ctag{background:#e8f0fe;color:#1967d2;border-radius:4px;padding:3px 9px;font-size:11px}

/* ── Tooltip ── */
#tt{position:fixed;display:none;background:#16213e;color:#e8f0fe;
    font-size:12px;line-height:1.6;padding:9px 13px;border-radius:7px;
    pointer-events:none;z-index:200;max-width:380px;box-shadow:0 4px 12px rgba(0,0,0,.3)}
</style>
</head>
<body>

<div class="tab-bar" id="tabBar"></div>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-hdr" id="sidebarHdr">Lattices</div>
    <div class="query-list" id="queryList"></div>
  </div>
  <div class="content" id="content">
    <div class="placeholder">Select a lattice from the sidebar</div>
  </div>
</div>
<div id="tt"></div>

<script>
const DATA = __DATA__;
const COLORS = ['#4e79a7','#f28e2b','#59a14f','#e15759','#76b7b2'];
const DS_KEYS = Object.keys(DATA);
let curDs  = DS_KEYS[0];
let curQid = null;
let curLat = null;
let selectedEraLayer = null;
let currentEraNodes  = [];

// ── Tabs ──────────────────────────────────────────────────────────────────
function buildTabs() {
  document.getElementById('tabBar').innerHTML = DS_KEYS.map(k =>
    `<div class="tab${k===curDs?' active':''}" onclick="selDs('${k}')">
       ${DATA[k].label}
       <span style="opacity:.55;font-size:11px;margin-left:4px">(${DATA[k].lattices.length})</span>
     </div>`
  ).join('');
}

function selDs(k) {
  curDs=k; curQid=null; curLat=null; selectedEraLayer=null; currentEraNodes=[];
  buildTabs(); buildSidebar();
  document.getElementById('content').innerHTML='<div class="placeholder">Select a lattice</div>';
}

// ── Sidebar ───────────────────────────────────────────────────────────────
function buildSidebar() {
  const d = DATA[curDs];
  document.getElementById('sidebarHdr').textContent = `${d.lattices.length} passing lattices`;
  document.getElementById('queryList').innerHTML = d.lattices.map(l =>
    `<div class="qitem${l.qid===curQid?' active':''}" onclick="selQuery('${l.qid}')">
       <div class="qitem-id">QID ${l.qid}</div>
       <div class="qitem-q">${l.question||'(no question text)'}</div>
     </div>`
  ).join('');
}

// ── Query render ──────────────────────────────────────────────────────────
function selQuery(qid) {
  curQid = qid;
  curLat = DATA[curDs].lattices.find(l => l.qid === qid);
  selectedEraLayer = null;
  currentEraNodes  = [];
  buildSidebar();
  renderContent(curLat);
}

function renderContent(lat) {
  const isP = curDs.includes('pragma');
  const dim  = isP ? 'pragma' : 'criterion';
  const dims = isP ? 'pragmas' : 'criteria';

  const answersHtml = lat.answers.map((a,i) =>
    `<div class="achip">
       <div class="aswatch" style="background:${COLORS[i]}"></div>
       <div>${a}</div>
     </div>`
  ).join('');

  const removedN    = lat.criteria.length - lat.bfs_layer;
  const removedHtml = lat.bfs_removed.map(r=>`<span class="rtag">${r}</span>`).join('');
  const critsHtml   = lat.criteria.map(c=>`<span class="ctag">${c}</span>`).join('');

  document.getElementById('content').innerHTML = `
    <div class="q-text">${lat.question||'QID '+lat.qid}</div>

    <div class="sec">Answers</div>
    <div class="answers">${answersHtml}</div>

    <div class="bfs-banner">
      <strong>BFS counterfactual</strong> at layer <strong>${lat.bfs_layer}</strong>
      — ${removedN} ${removedN===1?dim:dims} removed: ${removedHtml}
    </div>

    <div class="chart-card">
      <div class="chart-title">Layer distribution — proportion of nodes with each top-1 answer</div>
      <div class="chart-sub">Click a layer bar (2–9) to see ERA Phase 1 entropy analysis ↓</div>
      <div class="chart-wrap" id="chartWrap"></div>
    </div>

    <div id="era-panel" style="display:none">
      <div class="chart-card">
        <div class="chart-title" id="era-title"></div>
        <div id="era-chart" style="overflow-x:auto"></div>
        <div class="legend" id="era-legend" style="margin-top:10px"></div>
      </div>
    </div>

    <div class="sec">${dims.charAt(0).toUpperCase()+dims.slice(1)}</div>
    <div class="ctags">${critsHtml}</div>
  `;

  renderChart(lat);
}

// ── Distribution chart ────────────────────────────────────────────────────
function renderChart(lat) {
  const W=680, H=290, ml=44, mr=24, mt=46, mb=56;
  const cW=W-ml-mr, cH=H-mt-mb;
  const nL=11, gap=5;
  const bW = cW/nL - gap;

  const parts = [`<svg width="${W}" height="${H}" style="font-family:inherit;display:block;">`];
  parts.push(`<g transform="translate(${ml},${mt})">`);

  // Y-axis label
  parts.push(`<text transform="rotate(-90)" x="${-cH/2}" y="-34" text-anchor="middle"
    font-size="11" fill="#aaa">% of nodes at layer</text>`);

  // Grid lines + y ticks
  [0,25,50,75,100].forEach(p=>{
    const y = cH*(1-p/100);
    parts.push(`<line x1="0" y1="${y}" x2="${cW}" y2="${y}" stroke="#ececec" stroke-width="1"/>`);
    parts.push(`<text x="-7" y="${y+4}" text-anchor="end" font-size="10" fill="#bbb">${p}%</text>`);
  });

  // Column groups (clickable for ERA layers 2–9)
  lat.layers.forEach((counts, li) => {
    const total = counts.reduce((a,b)=>a+b,0);
    const x = li*(cW/nL) + gap/2;
    const isCF = li === lat.bfs_layer;
    const isEraLayer = li >= 2 && li <= lat.criteria.length - 1;
    const cursor = isEraLayer ? 'pointer' : 'default';

    parts.push(`<g id="col-${li}" onclick="${isEraLayer?'selectEraLayer('+li+')':''}" style="cursor:${cursor}">`);

    // Selection highlight (hidden by default)
    parts.push(`<rect id="hlcol-${li}" x="${(x-2).toFixed(1)}" y="${(-mt+4).toFixed(1)}"
      width="${(bW+4).toFixed(1)}" height="${(cH+mt-4).toFixed(1)}"
      fill="#4e79a7" rx="3" opacity="0" pointer-events="none"/>`);

    // CF highlight background (yellow)
    if (isCF) {
      parts.push(`<rect x="${(x-2).toFixed(1)}" y="${(-mt+4).toFixed(1)}" width="${(bW+4).toFixed(1)}" height="${(cH+mt-4).toFixed(1)}"
        fill="#fffbe6" rx="3" pointer-events="none"/>`);
    }

    if (total === 0) {
      parts.push(`<rect x="${x}" y="0" width="${bW}" height="${cH}" fill="#f4f4f4" rx="2"/>`);
    } else {
      let sy = cH;
      counts.forEach((c,ai)=>{
        if (!c) return;
        const h = (c/total)*cH;
        sy -= h;
        parts.push(`<rect x="${x.toFixed(1)}" y="${sy.toFixed(2)}" width="${bW.toFixed(1)}" height="${h.toFixed(2)}"
          fill="${COLORS[ai]}" opacity="0.88" rx="1"
          onmouseenter="showTT(event,${li},${total})"
          onmouseleave="hideTT()"/>`);
      });
    }

    // X-axis layer label
    parts.push(`<text x="${(x+bW/2).toFixed(1)}" y="${cH+17}" text-anchor="middle"
      font-size="11" fill="${isCF?'#c67d00':isEraLayer?'#4e79a7':'#999'}"
      font-weight="${isCF?'700':isEraLayer?'500':'400'}">${li}</text>`);

    // Node count below
    if (total > 0) {
      parts.push(`<text x="${(x+bW/2).toFixed(1)}" y="${cH+32}" text-anchor="middle"
        font-size="9" fill="#ccc">(${total})</text>`);
    }

    // CF arrow + label above
    if (isCF) {
      parts.push(`<text x="${(x+bW/2).toFixed(1)}" y="-24" text-anchor="middle" font-size="10"
        fill="#c67d00" font-weight="600">CF</text>`);
      parts.push(`<text x="${(x+bW/2).toFixed(1)}" y="-12" text-anchor="middle" font-size="13"
        fill="#c67d00">▼</text>`);
    }

    parts.push('</g>');
  });

  // X-axis label
  const xLabel = curDs.includes('pragma') ? 'Layer (number of pragmas)' : 'Layer (number of criteria)';
  parts.push(`<text x="${cW/2}" y="${cH+50}" text-anchor="middle"
    font-size="12" fill="#888">${xLabel}</text>`);
  parts.push(`<line x1="0" y1="${cH}" x2="${cW}" y2="${cH}" stroke="#ddd" stroke-width="1"/>`);

  parts.push('</g></svg>');

  // Legend (HTML)
  const legendHtml = `<div class="legend">${
    lat.answers.map((a,i)=>
      `<div class="leg-item">
         <div class="leg-dot" style="background:${COLORS[i]}"></div>
         <span title="${a}">${a.length>50?a.slice(0,50)+'…':a}</span>
       </div>`
    ).join('')
  }</div>`;

  document.getElementById('chartWrap').innerHTML = parts.join('') + legendHtml;
}

// ── ERA layer selection ───────────────────────────────────────────────────
function selectEraLayer(li) {
  if (selectedEraLayer !== null) {
    const prev = document.getElementById('hlcol-' + selectedEraLayer);
    if (prev) prev.setAttribute('opacity', '0');
  }
  const curr = document.getElementById('hlcol-' + li);
  if (curr) curr.setAttribute('opacity', '0.12');
  selectedEraLayer = li;
  renderEraPanel(curLat, li);
}

// ── ERA entropy panel ─────────────────────────────────────────────────────
// ERA dot fill = evaluation status; background bar = actual top-1 answer
function nodeStatusColor(flags) {
  if (flags & 8) return '#4e79a7';  // BFS CF (blue)
  if (flags & 4) return '#e15759';  // ERA CF (red)
  if (flags & 2) return '#59a14f';  // P2 evaluated (green)
  if (flags & 1) return '#e6a817';  // P1 selected (amber)
  return '#b0b8c4';                 // not evaluated (gray)
}

function renderEraPanel(lat, li) {
  const panel    = document.getElementById('era-panel');
  const eraChart = document.getElementById('era-chart');
  const eraTitle = document.getElementById('era-title');
  const eraLeg   = document.getElementById('era-legend');

  const eraData = lat.era && lat.era[li];
  if (!eraData) {
    panel.style.display = 'none';
    return;
  }

  const nodes = eraData.nodes;   // [[bitmask, entropy, flags], ...]
  const cut   = eraData.cut;     // ideal cutoff index
  const n     = nodes.length;

  currentEraNodes = nodes;

  const nP1      = nodes.filter(nd => nd[2] & 1).length;
  const nP2      = nodes.filter(nd => nd[2] & 2).length;
  const hasEraCF = nodes.some(nd => nd[2] & 4);
  const hasBfsCF = nodes.some(nd => nd[2] & 8);

  const eraLayer = nodes.find(nd => nd[2] & 4);
  const bfsLayer = nodes.find(nd => nd[2] & 8);
  let cfNote = '';
  if (hasEraCF) cfNote += ' · <span style="color:#e15759">ERA CF in this layer</span>';
  if (hasBfsCF) cfNote += ' · <span style="color:#4e79a7">BFS CF in this layer</span>';

  eraTitle.innerHTML =
    `Layer ${li} — ERA Phase 1 entropy (softmax, max-jump, budget=4)`+
    ` · ${n} nodes · ${nP1} P1 selected · ${nP2} P2 evaluated${cfNote}`;
  panel.style.display = 'block';

  // SVG dimensions
  const W=720, H=210, ml=54, mr=20, mt=28, mb=38;
  const cW=W-ml-mr, cH=H-mt-mb;
  const maxEnt = nodes.reduce((m, nd) => Math.max(m, nd[1]), 0) || 1;

  // X position: spread dots from xPad to cW-xPad so edge dots aren't clipped
  const xPad = 8;
  const xPos = i => n > 1 ? xPad + i * (cW - 2*xPad) / (n - 1) : cW / 2;
  const dotR = n > 150 ? 2.5 : n > 60 ? 3 : n > 20 ? 4 : 5.5;

  const parts = [`<svg width="${W}" height="${H}" style="font-family:inherit;display:block">`];
  parts.push(`<g transform="translate(${ml},${mt})">`);

  // Y-axis label
  parts.push(`<text transform="rotate(-90)" x="${-cH/2}" y="-42" text-anchor="middle" font-size="10" fill="#aaa">Softmax Entropy</text>`);

  // Grid + Y ticks
  [0, 0.5, 1.0].forEach(p => {
    const yv = (maxEnt * p).toFixed(3);
    const y  = cH * (1 - p);
    parts.push(`<line x1="0" y1="${y.toFixed(1)}" x2="${cW}" y2="${y.toFixed(1)}" stroke="#ececec"/>`);
    parts.push(`<text x="-4" y="${(y+3.5).toFixed(1)}" text-anchor="end" font-size="9" fill="#bbb">${yv}</text>`);
  });

  // Background bars — actual top-1 answer color, one per node
  const bgBarW = n > 1 ? (cW - 2*xPad) / (n - 1) : cW;
  nodes.forEach((nd, i) => {
    const [bitmask, ent, flags, actual_ans_idx] = nd;
    if (actual_ans_idx < 0) return;
    const cx  = xPos(i);
    const x0  = Math.max(0, cx - bgBarW / 2).toFixed(1);
    const w   = Math.min(bgBarW, cW - parseFloat(x0)).toFixed(1);
    parts.push(`<rect x="${x0}" y="0" width="${w}" height="${cH}" fill="${COLORS[actual_ans_idx]}" opacity="0.2" pointer-events="none"/>`);
  });

  // Connecting line (drawn above background)
  if (n > 1) {
    const pts = nodes.map((nd, i) => {
      const x = xPos(i).toFixed(1);
      const y = (cH - (nd[1] / maxEnt) * cH).toFixed(1);
      return `${x},${y}`;
    }).join(' ');
    parts.push(`<polyline points="${pts}" fill="none" stroke="#8898aa" stroke-width="1.5"/>`);
  }

  // Ideal cutoff vertical line
  if (n > 1 && cut < n - 1) {
    const cx = ((xPos(cut) + xPos(cut + 1)) / 2).toFixed(1);
    parts.push(`<line x1="${cx}" y1="${(-mt+8).toFixed(1)}" x2="${cx}" y2="${cH}"
      stroke="#c67d00" stroke-width="1.5" stroke-dasharray="5,3"/>`);
    parts.push(`<text x="${cx}" y="${(-mt+6).toFixed(1)}" text-anchor="middle"
      font-size="8" fill="#c67d00">ideal cut</text>`);
  }

  // Dots — colored by evaluation status, drawn on top
  nodes.forEach((nd, i) => {
    const [bitmask, ent, flags, actual_ans_idx] = nd;
    const x      = xPos(i).toFixed(1);
    const y      = (cH - (ent / maxEnt) * cH).toFixed(1);
    const dotCol = nodeStatusColor(flags);
    const hitR   = Math.max(dotR + 2, 5);
    parts.push(`<circle cx="${x}" cy="${y}" r="${hitR}" fill="transparent"
      onmouseenter="showEraTT(event,${i})" onmouseleave="hideTT()"/>`);
    parts.push(`<circle cx="${x}" cy="${y}" r="${dotR}" fill="${dotCol}" stroke="white" stroke-width="0.8" pointer-events="none"/>`);
  });

  // X axis
  parts.push(`<line x1="0" y1="${cH}" x2="${cW}" y2="${cH}" stroke="#ddd"/>`);
  parts.push(`<text x="${cW/2}" y="${cH+28}" text-anchor="middle" font-size="10" fill="#888">Nodes sorted by entropy (desc) — index 0 = highest entropy</text>`);

  parts.push('</g></svg>');
  eraChart.innerHTML = parts.join('');

  // Legend
  const ansLeg = lat.answers.map((a, i) =>
    `<div class="leg-item">
       <div class="leg-dot" style="background:${COLORS[i]};opacity:0.5;border-radius:2px"></div>
       <span title="${a}">${a.length>44?a.slice(0,44)+'…':a}</span>
     </div>`
  ).join('');
  const statusDefs = [
    ['#e6a817', `P1 selected (${nP1})`],
    ['#59a14f', `P2 evaluated (${nP2})`],
    hasEraCF ? ['#e15759', 'ERA CF']  : null,
    hasBfsCF ? ['#4e79a7', 'BFS CF']  : null,
    ['#b0b8c4', 'not evaluated'],
  ].filter(Boolean);
  const statusLeg = statusDefs.map(([c, l]) =>
    `<div class="leg-item">
       <div class="leg-dot" style="background:${c};border-radius:50%"></div>
       <span>${l}</span>
     </div>`
  ).join('');
  eraLeg.innerHTML =
    `<div style="font-size:10px;font-weight:700;color:#9aa5b4;letter-spacing:.6px;text-transform:uppercase;margin-bottom:5px">Background — actual top-1 answer</div>`+
    `<div class="legend" style="margin-bottom:10px">${ansLeg}</div>`+
    `<div style="font-size:10px;font-weight:700;color:#9aa5b4;letter-spacing:.6px;text-transform:uppercase;margin-bottom:5px">Dot — evaluation status</div>`+
    `<div class="legend">${statusLeg}</div>`;
}

// ── Tooltips ──────────────────────────────────────────────────────────────
function showTT(event, layer, total) {
  if (!curLat) return;
  const counts  = curLat.layers[layer];
  const answers = curLat.answers;
  const lines   = [`<strong>Layer ${layer}</strong> &nbsp;(${total} node${total!==1?'s':''})`];
  counts.forEach((c,i)=>{
    if (!c) return;
    const pct = (c/total*100).toFixed(1);
    lines.push(`<span style="color:${COLORS[i]}">■</span> `+
      `${answers[i].length>40?answers[i].slice(0,40)+'…':answers[i]}: `+
      `<strong>${c}</strong> (${pct}%)`);
  });
  const tt = document.getElementById('tt');
  tt.innerHTML = lines.join('<br>');
  tt.style.display = 'block';
  moveTT(event);
}

function showEraTT(event, idx) {
  const nd = currentEraNodes[idx];
  if (!nd || !curLat) return;
  const [bitmask, ent, flags, actual_ans_idx] = nd;
  const subset   = curLat.criteria.filter((_, i) => bitmask & (1 << i));
  const ansName  = actual_ans_idx >= 0 ? curLat.answers[actual_ans_idx] : '?';
  const ansColor = actual_ans_idx >= 0 ? COLORS[actual_ans_idx] : '#ccc';
  const statuses = [];
  if (flags & 8) statuses.push(`<span style="color:#4e79a7">BFS CF</span>`);
  if (flags & 4) statuses.push(`<span style="color:#e15759">ERA CF</span>`);
  if (flags & 2) statuses.push(`<span style="color:#59a14f">P2 evaluated</span>`);
  if (flags & 1) statuses.push(`<span style="color:#e6a817">P1 selected</span>`);
  if (!statuses.length) statuses.push('not evaluated');
  const tt = document.getElementById('tt');
  tt.innerHTML = [
    `<strong>Node ${idx+1}/${currentEraNodes.length}</strong> · entropy = ${ent.toFixed(4)}`,
    `Top-1 (actual): <span style="color:${ansColor}">■</span> ${ansName.length>50?ansName.slice(0,50)+'…':ansName}`,
    `Status: ${statuses.join(', ')}`,
    `Subset (${subset.length}): ${subset.join(', ')}`,
  ].join('<br>');
  tt.style.display = 'block';
  moveTT(event);
}

function hideTT() { document.getElementById('tt').style.display='none'; }

document.addEventListener('mousemove', e => {
  const tt = document.getElementById('tt');
  if (tt.style.display === 'block') moveTT(e);
});

function moveTT(e) {
  const tt = document.getElementById('tt');
  const x  = Math.min(e.clientX+14, window.innerWidth-tt.offsetWidth-8);
  const y  = Math.max(e.clientY-8, 4);
  tt.style.left = x+'px';
  tt.style.top  = y+'px';
}

// ── Init ──────────────────────────────────────────────────────────────────
buildTabs();
buildSidebar();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate() -> None:
    all_data = build_all_data()
    data_json = json.dumps(all_data, ensure_ascii=False)
    html = HTML.replace("__DATA__", data_json)
    os.makedirs("artifacts", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(OUT_PATH) // 1024
    print(f"\nWrote {OUT_PATH}  ({size_kb} KB)")


if __name__ == "__main__":
    generate()
