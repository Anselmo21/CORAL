"""For ERA's negative-gap cases: at the layer ERA stopped on (L_proxy),
does there exist ANY valid CF that is a subset of BFS's CF?

Interpretation: if such a subset-CF exists, ERA could in principle have found
it with more exploration at that layer — its early termination missed a CF
within BFS's own branch. If no such CF exists at L_proxy, the layer is
fundamentally outside BFS's branch and more budget would not help.

Buckets per case:
  - already_subset      : ERA's CF is already ⊂ BFS's CF (no exploration needed).
  - reachable           : ERA's CF is NOT ⊂ BFS's CF, but ∃ another CF at L_proxy
                          that IS ⊂ BFS's CF — extra budget could recover it.
  - unreachable         : ERA's CF is NOT ⊂ BFS's CF, and NO CF at L_proxy lies
                          inside BFS's CF — that layer offers no path back.

Writes:
    temp_plots/era_layer_reachability.txt
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from search.lattice import combo_key, parse_combo_key

DATASETS = [
    ("msmarco_criteria", "data/llama31-8b/msmarco-criteria-lattices.json", "MSMARCO criteria"),
    ("curated_criteria", "data/llama31-8b/curated-criteria-lattices.json", "Curated criteria"),
    ("msmarco_pragma",   "data/llama31-8b/msmarco-pragmas-lattices.json",  "MSMARCO pragma"),
    ("curated_pragma",   "data/llama31-8b/curated-pragmas-lattices.json",  "Curated pragma"),
]

METHODS = ["era_b4", "era_b8"]

OUT_DIR = "temp_plots"


def _load_lattices(path: str) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for qid, lat in raw.items():
        if qid == "_meta":
            continue
        crit = list(lat.get("criteria") or [])
        rbc = lat.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _analyze_dataset(name: str, data_path: str, nice: str, log_lines: list[str]):
    with open(f"logged_results/llama31-8b/{name}/paper_results.json") as f:
        rows = json.load(f)
    lattices = _load_lattices(data_path)

    stats = defaultdict(lambda: {"neg": 0, "already_subset": 0,
                                 "reachable": 0, "unreachable": 0})

    for row in rows:
        qid = row["qid"]
        start = row["start_node"]
        criteria, rbc = lattices[qid]
        sorted_full = sorted(criteria)
        bfs_layer = row["bfs"]["layer"]
        bfs_key = row["bfs"]["key"]
        bfs_set = frozenset(parse_combo_key(bfs_key, sorted_full)) if bfs_key else frozenset()
        start_key = combo_key(start)
        baseline_top1 = (rbc.get(start_key) or [""])[0]

        for m in METHODS:
            cell = row.get(m)
            if not cell:
                continue
            p_layer = cell.get("proxy_layer")
            p_key = cell.get("proxy_key")
            if p_layer is None or p_key is None:
                continue
            if p_layer >= bfs_layer:
                continue  # positive/zero gap — out of scope

            p_set = frozenset(parse_combo_key(p_key, sorted_full))
            stats[m]["neg"] += 1

            # (1) Already a subset of BFS's CF?
            if bfs_set and p_set < bfs_set:
                stats[m]["already_subset"] += 1
                continue

            # (2) Is there ANY CF at layer p_layer that is ⊂ BFS's CF?
            #     Only enumerate subsets of bfs_set of size p_layer.
            if not bfs_set or p_layer > len(bfs_set):
                stats[m]["unreachable"] += 1
                continue

            found = False
            for sub in combinations(sorted(bfs_set), p_layer):
                k = combo_key(sub)
                ranking = rbc.get(k)
                if ranking and ranking[0] != baseline_top1:
                    found = True
                    break
            if found:
                stats[m]["reachable"] += 1
            else:
                stats[m]["unreachable"] += 1

    log_lines.append(f"\n=== {nice} ({name}) ===")
    log_lines.append(f"{'method':<8} {'neg':>5} "
                     f"{'already ⊂ BFS':>16} "
                     f"{'reachable (∃ CF ⊂ BFS @ L_proxy)':>36} "
                     f"{'unreachable':>14}")
    for m in METHODS:
        s = stats[m]
        if s["neg"] == 0:
            log_lines.append(f"{m:<8} {0:>5}")
            continue
        pct_a = 100.0 * s["already_subset"] / s["neg"]
        pct_r = 100.0 * s["reachable"] / s["neg"]
        pct_u = 100.0 * s["unreachable"] / s["neg"]
        log_lines.append(
            f"{m:<8} {s['neg']:>5} "
            f"{s['already_subset']:>6} ({pct_a:>5.1f}%)    "
            f"{s['reachable']:>6} ({pct_r:>5.1f}%)                    "
            f"{s['unreachable']:>6} ({pct_u:>5.1f}%)"
        )
    return stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines: list[str] = []
    overall = defaultdict(lambda: {"neg": 0, "already_subset": 0,
                                   "reachable": 0, "unreachable": 0})
    for name, data_path, nice in DATASETS:
        per = _analyze_dataset(name, data_path, nice, log_lines)
        for m, s in per.items():
            for k in s:
                overall[m][k] += s[k]

    log_lines.append("\n=== ALL DATASETS COMBINED ===")
    log_lines.append(f"{'method':<8} {'neg':>5} "
                     f"{'already ⊂ BFS':>16} "
                     f"{'reachable (∃ CF ⊂ BFS @ L_proxy)':>36} "
                     f"{'unreachable':>14}")
    for m in METHODS:
        s = overall[m]
        if s["neg"] == 0:
            log_lines.append(f"{m:<8} {0:>5}")
            continue
        pct_a = 100.0 * s["already_subset"] / s["neg"]
        pct_r = 100.0 * s["reachable"] / s["neg"]
        pct_u = 100.0 * s["unreachable"] / s["neg"]
        log_lines.append(
            f"{m:<8} {s['neg']:>5} "
            f"{s['already_subset']:>6} ({pct_a:>5.1f}%)    "
            f"{s['reachable']:>6} ({pct_r:>5.1f}%)                    "
            f"{s['unreachable']:>6} ({pct_u:>5.1f}%)"
        )

    text = "\n".join(log_lines)
    out_path = os.path.join(OUT_DIR, "era_layer_reachability.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n → wrote {out_path}")


if __name__ == "__main__":
    main()
