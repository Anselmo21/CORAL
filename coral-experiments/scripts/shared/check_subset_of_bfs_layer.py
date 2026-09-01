"""For proxy cases with negative signed layer gap (proxy_layer < bfs_layer),
check whether the proxy's CF is a subset of SOME valid CF at the BFS layer.

Interpretation: BFS returned the shallowest CF at layer L_bfs. The proxy went
deeper (fewer criteria). We ask — is the proxy's smaller CF a strict subset of
any *other* valid CF sitting at layer L_bfs? If yes, the proxy is under a
shallower CF; if no, the proxy found a CF in a different branch entirely.

Writes:
    temp_plots/subset_of_bfs_layer.txt   (per-dataset counts + overall)
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

METHODS = ["sra", "cc_b4", "cc_b8", "era_b4", "era_b8"]

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

    per_method_totals = defaultdict(lambda: {"neg": 0, "subset_of_cf": 0,
                                             "proper_subset_of_bfs": 0,
                                             "any_superset_available": 0})

    for row in rows:
        qid = row["qid"]
        start = row["start_node"]
        start_set = frozenset(start)
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
            gap = p_layer - bfs_layer
            if gap >= 0:
                continue

            p_set = frozenset(parse_combo_key(p_key, sorted_full))
            if not (p_set <= start_set):
                continue  # sanity: proxy CF must live in start_node's lattice

            per_method_totals[m]["neg"] += 1

            # Is proxy set a subset of the specific BFS CF?
            if bfs_set and p_set < bfs_set:
                per_method_totals[m]["proper_subset_of_bfs"] += 1

            # Enumerate all supersets of p_set within start_node at size bfs_layer.
            remaining = sorted(start_set - p_set)
            needed = bfs_layer - len(p_set)
            found_valid_cf_superset = False
            any_superset_present = False
            if needed >= 0 and needed <= len(remaining):
                for extra in combinations(remaining, needed):
                    super_set = p_set | set(extra)
                    k = combo_key(super_set)
                    ranking = rbc.get(k)
                    if ranking is None:
                        continue
                    any_superset_present = True
                    if ranking[0] != baseline_top1:
                        found_valid_cf_superset = True
                        break
            if any_superset_present:
                per_method_totals[m]["any_superset_available"] += 1
            if found_valid_cf_superset:
                per_method_totals[m]["subset_of_cf"] += 1

    log_lines.append(f"\n=== {nice} ({name}) ===")
    log_lines.append(f"{'method':<8} {'neg cases':>10} {'⊂ some CF@BFS layer':>22} "
                     f"{'(⊂ BFS CF itself)':>20} {'supersets rbc-populated':>26}")
    for m in METHODS:
        s = per_method_totals[m]
        if s["neg"] == 0:
            log_lines.append(f"{m:<8} {0:>10}")
            continue
        pct = 100.0 * s["subset_of_cf"] / s["neg"]
        pct_bfs = 100.0 * s["proper_subset_of_bfs"] / s["neg"]
        pct_pop = 100.0 * s["any_superset_available"] / s["neg"]
        log_lines.append(
            f"{m:<8} {s['neg']:>10} "
            f"{s['subset_of_cf']:>6} ({pct:>5.1f}%)      "
            f"{s['proper_subset_of_bfs']:>6} ({pct_bfs:>5.1f}%)   "
            f"{s['any_superset_available']:>6} ({pct_pop:>5.1f}%)"
        )
    return per_method_totals


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines: list[str] = []
    overall = defaultdict(lambda: {"neg": 0, "subset_of_cf": 0,
                                   "proper_subset_of_bfs": 0,
                                   "any_superset_available": 0})
    for name, data_path, nice in DATASETS:
        per = _analyze_dataset(name, data_path, nice, log_lines)
        for m, s in per.items():
            for k in s:
                overall[m][k] += s[k]

    log_lines.append("\n=== ALL DATASETS COMBINED ===")
    log_lines.append(f"{'method':<8} {'neg cases':>10} {'⊂ some CF@BFS layer':>22} "
                     f"{'(⊂ BFS CF itself)':>20} {'supersets rbc-populated':>26}")
    for m in METHODS:
        s = overall[m]
        if s["neg"] == 0:
            log_lines.append(f"{m:<8} {0:>10}")
            continue
        pct = 100.0 * s["subset_of_cf"] / s["neg"]
        pct_bfs = 100.0 * s["proper_subset_of_bfs"] / s["neg"]
        pct_pop = 100.0 * s["any_superset_available"] / s["neg"]
        log_lines.append(
            f"{m:<8} {s['neg']:>10} "
            f"{s['subset_of_cf']:>6} ({pct:>5.1f}%)      "
            f"{s['proper_subset_of_bfs']:>6} ({pct_bfs:>5.1f}%)   "
            f"{s['any_superset_available']:>6} ({pct_pop:>5.1f}%)"
        )

    text = "\n".join(log_lines)
    out_path = os.path.join(OUT_DIR, "subset_of_bfs_layer.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n → wrote {out_path}")


if __name__ == "__main__":
    main()
