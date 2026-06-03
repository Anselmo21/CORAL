#!/usr/bin/env python3
"""Generate lattice_view_dev.html — one SVG per dev.jsonl case, grouped by k.

Each card shows the *effective* lattice of a case (toggleable subsets with
fixed criteria pinned). Includes layer 0 (the empty toggleable subset) so the
case's only CF, if any, can be seen even when it sits at the lattice floor.

Visual encoding:
  - Node fill        = top-1 answer at that effective subset (consistent palette).
  - Thick black ring = starting node (full toggleable for top-down, empty for bottom-up).
  - Yellow ring      = verified counterfactual (top-1 != baseline_top1).
  - Card header      = Q{qid} · {direction} · k={k}.
  - Card tooltip     = fixed criteria + baseline top-1.

Run:
    conda run -n coral python gen_dev_lattice_view.py
"""

import html as html_lib
import json
from itertools import combinations


LATTICES_PATH = "data/all_lattices.json"
CASES_PATH = "data/dev.jsonl"
OUTPUT = "lattice_view_dev.html"


def combo_key(subset: list[str]) -> str:
    return ", ".join(sorted(subset))


# ── Load data ─────────────────────────────────────────────────────────────

with open(LATTICES_PATH) as f:
    lattices = json.load(f)["lattices"]

with open(CASES_PATH) as f:
    cases = [json.loads(L) for L in f]

# Collect every top-1 across the effective lattices we'll show, to size palette
all_answers: set[str] = set()
for rec in cases:
    lat = lattices.get(str(rec["query_id"]))
    if lat is None:
        continue
    fixed = list(rec["fixed_criteria"])
    toggleable = list(rec["toggleable_criteria"])
    rk = lat["rankings"]
    for r in range(len(toggleable) + 1):
        for sub in combinations(toggleable, r):
            full_key = combo_key(list(sub) + fixed)
            ranking = rk.get(full_key, [])
            if ranking:
                all_answers.add(ranking[0])


PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#8dd3c7", "#ffffb3",
    "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69",
    "#bc80bd", "#ccebc5", "#ffed6f", "#1b9e77", "#d95f02",
]
answer_color: dict[str, str] = {
    ans: PALETTE[i % len(PALETTE)] for i, ans in enumerate(sorted(all_answers))
}


# ── SVG rendering ─────────────────────────────────────────────────────────


def build_effective_rbc(
    toggleable: list[str],
    fixed: list[str],
    full_rk: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Project full lattice rankings down to toggleable-keyed dict (empty key included)."""
    out: dict[str, list[str]] = {}
    for r in range(len(toggleable) + 1):
        for sub in combinations(toggleable, r):
            sub_list = list(sub)
            full_key = combo_key(sub_list + fixed)
            ranking = full_rk.get(full_key)
            if ranking is not None:
                out[combo_key(sub_list)] = ranking
    return out


def make_svg(
    toggleable: list[str],
    effective_rbc: dict[str, list[str]],
    start_key: str,
    baseline_top1: str | None,
) -> str:
    k = len(toggleable)

    # Enumerate nodes layer by layer, layer 0 (empty) at top of structure for
    # bottom-up readability; we'll flip visually below so larger subsets sit higher.
    layers: dict[int, list[tuple[str, str | None, list[str]]]] = {}
    for ls in range(0, k + 1):
        nodes = []
        for combo in combinations(toggleable, ls):
            key = combo_key(list(combo))
            top1 = (effective_rbc.get(key) or [None])[0]
            nodes.append((key, top1, list(combo)))
        layers[ls] = nodes

    max_nodes = max(len(v) for v in layers.values())

    if max_nodes <= 4:
        node_r, h_gap, v_gap = 10, 28, 42
    elif max_nodes <= 10:
        node_r, h_gap, v_gap = 8, 22, 38
    elif max_nodes <= 20:
        node_r, h_gap, v_gap = 6, 17, 34
    elif max_nodes <= 40:
        node_r, h_gap, v_gap = 5, 14, 30
    else:
        node_r, h_gap, v_gap = 4, 12, 28

    pad = node_r + 8
    svg_w = max(max_nodes * h_gap + 2 * pad, 80)
    svg_h = (k + 1) * v_gap + 2 * pad

    # y axis: layer k at top, layer 0 at bottom.
    pos: dict[str, tuple[float, float]] = {}
    for ls in range(0, k + 1):
        nodes = layers[ls]
        n = len(nodes)
        y = pad + (k - ls) * v_gap
        for i, (key, _, _) in enumerate(nodes):
            x = pad + (i + 0.5) * (svg_w - 2 * pad) / n
            pos[key] = (x, y)

    parts = [f'<svg width="{int(svg_w)}" height="{int(svg_h)}" xmlns="http://www.w3.org/2000/svg">']

    # Edges between adjacent layers
    for ls in range(1, k + 1):
        for key, _, combo in layers[ls]:
            x1, y1 = pos[key]
            for i in range(len(combo)):
                child_key = combo_key(combo[:i] + combo[i + 1:])
                if child_key in pos:
                    x2, y2 = pos[child_key]
                    parts.append(
                        f'<line x1="{x1:.0f}" y1="{y1:.0f}" '
                        f'x2="{x2:.0f}" y2="{y2:.0f}" '
                        f'stroke="#ccc" stroke-width="1"/>'
                    )

    # Nodes
    for ls in range(0, k + 1):
        for key, top1, combo in layers[ls]:
            x, y = pos[key]
            fill = answer_color.get(top1, "#eee") if top1 else "#eee"
            ranking = effective_rbc.get(key, [])
            ranking_lines = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ranking))
            label = key if key else "<empty toggleable>"
            tip = html_lib.escape(f"{label}\n{ranking_lines}")

            is_start = key == start_key
            is_cf = (
                baseline_top1 is not None
                and top1 is not None
                and top1 != baseline_top1
                and not is_start
            )

            if is_start:
                stroke, sw, extra_r = "#000", "2.5", 2
            elif is_cf:
                stroke, sw, extra_r = "#f4c200", "2.0", 1
            else:
                stroke, sw, extra_r = "#666", "0.8", 0

            parts.append(
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{node_r + extra_r}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}">'
                f'<title>{tip}</title></circle>'
            )

    parts.append("</svg>")
    return "".join(parts)


# ── Build sections (group by k, split direction) ──────────────────────────


cases_by_k_dir: dict[tuple[int, str], list[dict]] = {}
for rec in cases:
    cases_by_k_dir.setdefault((rec["k"], rec["direction"]), []).append(rec)


def render_card(rec: dict) -> str:
    qid = rec["query_id"]
    lat = lattices.get(str(qid))
    if lat is None:
        return ""
    fixed = sorted(rec["fixed_criteria"])
    toggleable = sorted(rec["toggleable_criteria"])
    direction = rec["direction"]
    rk = lat["rankings"]

    erbc = build_effective_rbc(toggleable, fixed, rk)

    # Start key in toggleable-only coordinates
    if direction == "top-down":
        start_key = combo_key(toggleable)
    else:  # bottom-up: start at empty toggleable
        start_key = ""

    baseline_top1 = rec.get("starting_node_top1")
    svg = make_svg(toggleable, erbc, start_key, baseline_top1)

    arrow = "▽" if direction == "top-down" else "△"
    fixed_html = html_lib.escape(", ".join(fixed)) if fixed else "<em>none</em>"
    tip = html_lib.escape(
        f"fixed: {', '.join(fixed) or '(none)'}\n"
        f"toggleable: {', '.join(toggleable)}\n"
        f"baseline_top1: {baseline_top1}"
    )

    unique_answers = len({v[0] for v in erbc.values() if v})
    if unique_answers == 1:
        ans_tag = '<span class="mono">1 ans</span>'
    else:
        ans_tag = f'<span class="nanswers">{unique_answers} ans</span>'

    n_cf = rec.get("n_counterfactuals_reachable", "?")
    min_hops = rec.get("min_hops_to_counterfactual", "?")

    return (
        f'<div class="card" title="{tip}">'
        f'<div class="card-hdr">Q{qid} {arrow} k={rec["k"]} · {ans_tag} · '
        f'CF={n_cf} @ {min_hops}</div>'
        f'<div class="fixed">fixed: {fixed_html}</div>'
        f'{svg}'
        f'</div>'
    )


sections: list[str] = []
for k in sorted({k for k, _ in cases_by_k_dir}):
    for direction in ("top-down", "bottom-up"):
        bucket = cases_by_k_dir.get((k, direction), [])
        if not bucket:
            continue
        cards = "".join(render_card(r) for r in bucket)
        sections.append(
            f'<section>'
            f'<h2>k={k} · {direction} '
            f'<span class="meta">({len(bucket)} cases · 2^{k}={2**k} nodes each)</span></h2>'
            f'<div class="row">{cards}</div>'
            f'</section>'
        )


legend_html = "".join(
    f'<span class="leg"><span class="swatch" style="background:{color}"></span>'
    f'{html_lib.escape(ans[:40])}</span>'
    for ans, color in sorted(answer_color.items())
)


page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dev Lattice Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f0f2f5; color: #222; padding: 24px; }}
  h1 {{ font-size: 1.3em; font-weight: 700; margin-bottom: 6px; }}
  .intro {{ color: #555; font-size: 0.85em; margin-bottom: 16px; }}
  h2 {{ font-size: 0.95em; font-weight: 600; margin: 32px 0 10px; color: #333; }}
  .meta {{ font-size: 0.85em; font-weight: 400; color: #777; }}
  .legend {{ background: #fff; border: 1px solid #dde; border-radius: 6px;
             padding: 12px 16px; margin-bottom: 24px; font-size: 0.8em; }}
  .legend strong {{ display: block; margin-bottom: 8px; color: #333; }}
  .leg {{ display: inline-flex; align-items: center; margin: 3px 12px 3px 0; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 50%; border: 1px solid #999;
             margin-right: 5px; flex-shrink: 0; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .card {{ background: #fff; border: 1px solid #dde; border-radius: 6px;
           padding: 8px 10px; }}
  .card-hdr {{ font-size: 0.72em; color: #555; margin-bottom: 4px; font-weight: 500; }}
  .fixed {{ font-size: 0.68em; color: #888; margin-bottom: 6px;
            max-width: 260px; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; }}
  .mono {{ font-size: 0.78em; color: #c0392b; font-weight: 600; }}
  .nanswers {{ font-size: 0.78em; color: #888; }}
  .marker-legend {{ background: #fff; border: 1px solid #dde; border-radius: 6px;
                    padding: 10px 14px; margin-bottom: 12px; font-size: 0.78em;
                    display: flex; gap: 20px; flex-wrap: wrap; }}
  .marker-legend span {{ display: inline-flex; align-items: center; }}
  .marker-legend svg {{ margin-right: 5px; }}
</style>
</head>
<body>
<h1>Dev Lattice Viewer</h1>
<div class="intro">
  Each card is one dev.jsonl case; SVG shows its effective toggleable lattice
  (fixed criteria pinned in every node). Layer 0 (empty toggleable) sits at the
  bottom. Hover a node for its full ranking.
</div>
<div class="marker-legend">
  <span><svg width="20" height="20"><circle cx="10" cy="10" r="7"
    fill="#eee" stroke="#000" stroke-width="2.5"/></svg> starting node</span>
  <span><svg width="20" height="20"><circle cx="10" cy="10" r="7"
    fill="#eee" stroke="#f4c200" stroke-width="2"/></svg> counterfactual (top-1 ≠ baseline)</span>
  <span><svg width="20" height="20"><circle cx="10" cy="10" r="7"
    fill="#eee" stroke="#666" stroke-width="0.8"/></svg> other node (same as baseline)</span>
</div>
<div class="legend">
  <strong>Answer colour key:</strong>
  {legend_html}
</div>
{"".join(sections)}
</body>
</html>"""

with open(OUTPUT, "w") as f:
    f.write(page)

print(f"Written {OUTPUT} ({len(cases)} cases across {len({(r['k'], r['direction']) for r in cases})} (k, direction) buckets)")
