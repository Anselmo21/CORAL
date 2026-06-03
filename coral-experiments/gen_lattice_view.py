#!/usr/bin/env python3
"""Throwaway script: generates lattice_view.html visualising 20 sampled lattices per effective size."""

import json
import random
import html as html_lib
from itertools import combinations

DATA_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"
OUTPUT = "lattice_view.html"
SEED = 42
TARGET = 20


def combo_key(subset: list[str]) -> str:
    return ", ".join(sorted(subset))


with open(DATA_PATH) as f:
    data = json.load(f)
runs = data["runs"]

rng = random.Random(SEED)

# Gather every top-1 answer that appears anywhere in the data
all_answers: set[str] = set()
for run in runs:
    for ranking in run["rankings_by_combo"].values():
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


def make_svg(start_node: list[str], rbc: dict[str, list[str]], max_layers: int) -> str:
    s = len(start_node)
    display_layers = min(s, max_layers)

    layers: dict[int, list[tuple[str, str | None, list[str]]]] = {}
    for ls in range(1, display_layers + 1):
        nodes = []
        for combo in combinations(start_node, ls):
            key = combo_key(list(combo))
            top1 = (rbc.get(key) or [None])[0]
            nodes.append((key, top1, list(combo)))
        layers[ls] = nodes

    max_nodes = max(len(v) for v in layers.values())

    if max_nodes <= 8:
        node_r, h_gap, v_gap = 10, 26, 46
    elif max_nodes <= 15:
        node_r, h_gap, v_gap = 7, 19, 40
    elif max_nodes <= 30:
        node_r, h_gap, v_gap = 5, 15, 35
    else:
        node_r, h_gap, v_gap = 4, 12, 30

    pad = node_r + 6
    svg_w = max(max_nodes * h_gap + 2 * pad, 80)
    svg_h = display_layers * v_gap + 2 * pad

    # Start node at bottom (layer s = large y), singletons at top (layer 1 = small y)
    pos: dict[str, tuple[float, float]] = {}
    for ls in range(1, display_layers + 1):
        nodes = layers[ls]
        n = len(nodes)
        y = pad + (display_layers - ls) * v_gap
        for i, (key, _, _) in enumerate(nodes):
            x = pad + (i + 0.5) * (svg_w - 2 * pad) / n
            pos[key] = (x, y)

    parts = [f'<svg width="{int(svg_w)}" height="{int(svg_h)}" xmlns="http://www.w3.org/2000/svg">']

    # Edges
    for ls in range(2, display_layers + 1):
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
    for ls in range(1, display_layers + 1):
        for key, top1, combo in layers[ls]:
            x, y = pos[key]
            fill = answer_color.get(top1, "#eee") if top1 else "#eee"
            ranking = rbc.get(key, [])
            ranking_lines = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ranking))
            tip = html_lib.escape(f"{key}\n{ranking_lines}")
            stroke = "#222" if ls == display_layers else "#555"
            sw = "1.5" if ls == display_layers else "0.8"
            parts.append(
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{node_r}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}">'
                f'<title>{tip}</title></circle>'
            )

    parts.append("</svg>")
    return "".join(parts)


def has_cf(start_node: list[str], rbc: dict[str, list[str]]) -> bool:
    start_key = combo_key(start_node)
    baseline = (rbc.get(start_key) or [None])[0]
    if baseline is None:
        return False
    return any(
        v[0] != baseline
        for k, v in rbc.items()
        if v and k != start_key
    )


sections: list[str] = []

for eff_size in range(3, 11):
    candidates: list[dict] = []
    for run in runs:
        criteria = run["criteria_used"]
        n = len(criteria)
        rbc = run["rankings_by_combo"]
        if n == eff_size:
            candidates.append({"start_node": list(criteria), "rbc": rbc, "qid": run["query_id"]})
        elif n > eff_size:
            for combo in combinations(criteria, eff_size):
                sub_keys: set[str] = set()
                for r in range(1, eff_size + 1):
                    for c in combinations(combo, r):
                        sub_keys.add(combo_key(list(c)))
                sub_rbc = {k: rbc[k] for k in sub_keys if k in rbc}
                candidates.append({"start_node": list(combo), "rbc": sub_rbc, "qid": run["query_id"]})

    rng.shuffle(candidates)
    valid = [c for c in candidates if has_cf(c["start_node"], c["rbc"])]
    sampled = valid[:TARGET]

    if not sampled:
        sections.append(
            f'<section><h2>Size {eff_size}</h2><p class="empty">No candidates available.</p></section>'
        )
        continue

    # Cap displayed layers so SVGs stay readable for large lattices
    max_layers = min(eff_size, 7)
    truncated = eff_size > max_layers

    cards: list[str] = []
    for c in sampled:
        svg = make_svg(c["start_node"], c["rbc"], max_layers)
        trunc_note = (
            f'<div class="trunc">top {max_layers}/{eff_size} layers shown</div>'
            if truncated else ""
        )
        unique_answers = len({v[0] for v in c["rbc"].values() if v})
        mono_tag = ' <span class="mono">1 answer</span>' if unique_answers == 1 else f' <span class="nanswers">{unique_answers} answers</span>'
        criteria_tip = html_lib.escape(", ".join(c["start_node"]))
        cards.append(
            f'<div class="card" title="{criteria_tip}">'
            f'<div class="card-hdr">Q{c["qid"]}{mono_tag}</div>'
            f'{svg}{trunc_note}</div>'
        )

    total_nodes = 2 ** eff_size - 1
    sections.append(
        f'<section>'
        f'<h2>Effective size {eff_size} '
        f'<span class="meta">({len(sampled)} shown · {len(valid)} with CF · {total_nodes} nodes each)</span></h2>'
        f'<div class="row">{"".join(cards)}</div>'
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
<title>Lattice Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f0f2f5; color: #222; padding: 24px; }}
  h1 {{ font-size: 1.3em; font-weight: 700; margin-bottom: 16px; }}
  h2 {{ font-size: 0.95em; font-weight: 600; margin: 32px 0 10px; color: #333; }}
  .meta {{ font-size: 0.85em; font-weight: 400; color: #777; }}
  .legend {{ background: #fff; border: 1px solid #dde; border-radius: 6px; padding: 12px 16px; margin-bottom: 24px; font-size: 0.8em; }}
  .legend strong {{ display: block; margin-bottom: 8px; color: #333; }}
  .leg {{ display: inline-flex; align-items: center; margin: 3px 12px 3px 0; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 50%; border: 1px solid #999; margin-right: 5px; flex-shrink: 0; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ background: #fff; border: 1px solid #dde; border-radius: 6px; padding: 8px 10px; }}
  .card-hdr {{ font-size: 0.72em; color: #999; margin-bottom: 4px; }}
  .trunc {{ font-size: 0.72em; color: #c07000; text-align: center; margin-top: 4px; }}
  .mono {{ font-size: 0.8em; color: #c0392b; font-weight: 600; }}
  .nanswers {{ font-size: 0.8em; color: #888; }}
  .empty {{ color: #999; font-size: 0.9em; padding: 8px 0; }}
</style>
</head>
<body>
<h1>Lattice Viewer — down search &nbsp;·&nbsp; node colour = top-1 answer</h1>
<div class="legend">
  <strong>Answer colour key:</strong>
  {legend_html}
</div>
{"".join(sections)}
</body>
</html>"""

with open(OUTPUT, "w") as f:
    f.write(page)

print(f"Written {OUTPUT}")
