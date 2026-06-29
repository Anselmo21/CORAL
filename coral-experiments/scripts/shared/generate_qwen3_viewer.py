"""Generate a self-contained HTML viewer for Qwen3 lattice layer distributions.

For each of the 4 datasets, for every size-10 lattice that passes the >=2
removal filter, computes a layer-by-layer (0=empty → 10=full) breakdown of
how many nodes at each layer have each top-1 answer.

Output: artifacts/qwen3_hasse_viewer.html
Usage:  conda run -n coral python3 -m scripts.shared.generate_qwen3_viewer
"""
from __future__ import annotations

import json
import os
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key

FILES = [
    ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "MSMARCO Criteria"),
    ("sonnet_criteria",  "data/qwen3-32b/sonnet-criteria-qwen3-32b.json",   "Sonnet Criteria"),
    ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",   "MSMARCO Pragma"),
    ("sonnet_pragma",    "data/qwen3-32b/sonnet-pragmas-qwen3-32b.json",    "Sonnet Pragma"),
]

MIN_REMOVED = 2
OUT_PATH    = "artifacts/qwen3_hasse_viewer.html"


def _key_to_layer_map(criteria: list[str]) -> dict[str, int]:
    m: dict[str, int] = {"": 0}
    for k in range(1, len(criteria) + 1):
        for combo in combinations(criteria, k):
            m[combo_key(list(combo))] = k
    return m


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

        answers   = list(v["plausible_answers"])
        a2i       = {a: i for i, a in enumerate(answers)}
        k2l       = _key_to_layer_map(criteria)

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

        lattices.append({
            "qid":         qid,
            "question":    v.get("question", ""),
            "criteria":    criteria,
            "answers":     answers,
            "bfs_layer":   res["size"],
            "bfs_removed": removed,
            "layers":      layer_counts,  # list[11] of list[n_answers]
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
<title>Qwen3-32B Lattice Viewer</title>
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
.chart-title{font-size:13px;font-weight:600;color:#555;margin-bottom:14px}
.chart-wrap{overflow-x:auto}

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
    pointer-events:none;z-index:200;max-width:320px;box-shadow:0 4px 12px rgba(0,0,0,.3)}
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
  curDs=k; curQid=null; curLat=null;
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
  buildSidebar();
  renderContent(curLat);
}

function renderContent(lat) {
  const isP = curDs.includes('pragma');
  const dim = isP ? 'pragma' : 'criterion';
  const dims = isP ? 'pragmas' : 'criteria';

  const answersHtml = lat.answers.map((a,i) =>
    `<div class="achip">
       <div class="aswatch" style="background:${COLORS[i]}"></div>
       <div>${a}</div>
     </div>`
  ).join('');

  const removedN = lat.criteria.length - lat.bfs_layer;
  const removedHtml = lat.bfs_removed.map(r=>`<span class="rtag">${r}</span>`).join('');

  const critsHtml = lat.criteria.map(c=>`<span class="ctag">${c}</span>`).join('');

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
      <div class="chart-wrap" id="chartWrap"></div>
    </div>

    <div class="sec">${dims.charAt(0).toUpperCase()+dims.slice(1)}</div>
    <div class="ctags">${critsHtml}</div>
  `;

  renderChart(lat);
}

// ── Chart ─────────────────────────────────────────────────────────────────
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

  // Bars
  lat.layers.forEach((counts, li) => {
    const total = counts.reduce((a,b)=>a+b,0);
    const x = li*(cW/nL) + gap/2;
    const isCF = li === lat.bfs_layer;

    // CF highlight background
    if (isCF) {
      parts.push(`<rect x="${x-2}" y="${-mt+4}" width="${bW+4}" height="${cH+mt-4}"
        fill="#fffbe6" rx="3"/>`);
    }

    if (total === 0) {
      parts.push(`<rect x="${x}" y="0" width="${bW}" height="${cH}" fill="#f4f4f4" rx="2"/>`);
    } else {
      let sy = cH;
      counts.forEach((c,ai)=>{
        if (!c) return;
        const h = (c/total)*cH;
        sy -= h;
        parts.push(`<rect x="${x}" y="${sy.toFixed(2)}" width="${bW}" height="${h.toFixed(2)}"
          fill="${COLORS[ai]}" opacity="0.88" rx="1"
          onmouseenter="showTT(event,${li},${total})"
          onmouseleave="hideTT()"
          style="cursor:default"/>`);
      });
    }

    // X-axis layer label
    parts.push(`<text x="${x+bW/2}" y="${cH+17}" text-anchor="middle"
      font-size="11" fill="${isCF?'#c67d00':'#999'}"
      font-weight="${isCF?'700':'400'}">${li}</text>`);

    // Node count below
    if (total > 0) {
      parts.push(`<text x="${x+bW/2}" y="${cH+32}" text-anchor="middle"
        font-size="9" fill="#ccc">(${total})</text>`);
    }

    // CF arrow + label above
    if (isCF) {
      parts.push(`<text x="${x+bW/2}" y="-24" text-anchor="middle" font-size="10"
        fill="#c67d00" font-weight="600">CF</text>`);
      parts.push(`<text x="${x+bW/2}" y="-12" text-anchor="middle" font-size="13"
        fill="#c67d00">▼</text>`);
    }
  });

  // X-axis label
  parts.push(`<text x="${cW/2}" y="${cH+50}" text-anchor="middle"
    font-size="12" fill="#888">Layer (number of concepts)</text>`);
  parts.push(`<line x1="0" y1="${cH}" x2="${cW}" y2="${cH}" stroke="#ddd" stroke-width="1"/>`);

  parts.push('</g></svg>');

  // Legend (HTML, not SVG)
  const legendHtml = lat.answers.map((a,i)=>
    `<div class="leg-item">
       <div class="leg-dot" style="background:${COLORS[i]}"></div>
       <span title="${a}">${a.length>50?a.slice(0,50)+'…':a}</span>
     </div>`
  ).join('');

  document.getElementById('chartWrap').innerHTML =
    parts.join('') +
    `<div class="legend">${legendHtml}</div>`;
}

// ── Tooltip ───────────────────────────────────────────────────────────────
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
