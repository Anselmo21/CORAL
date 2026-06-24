"""Audit the current sampling on the 15-lattice set (after excluding 3 qids).

Tells us, per effective size k, BEFORE we change anything:
  - how many sub-lattice cases came from each qid
  - what the sub-lattice BFS CF layer distribution looks like
  - how many distinct ranking classes the sub-lattice has

Output: reports/era_pragma_pruning/15lattice_audit.txt (plus stdout)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

EXCLUDE_QIDS = {"480543", "51587", "52412"}
RESULTS_PATH = "data/18_lattices_cf_layer2to8/results.json"
DATASET_PATH = "data/18_lattices_cf_layer2to8.json"


def main() -> None:
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    results = [r for r in results if r["qid"] not in EXCLUDE_QIDS]

    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    cf_layer_full = {qid: rec["first_cf_layer"]
                     for qid, rec in dataset["lattices"].items()
                     if qid not in EXCLUDE_QIDS}

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    print(f"After excluding {sorted(EXCLUDE_QIDS)}: {len(results)} cases across "
          f"{len(set(r['qid'] for r in results))} qids.\n")

    # 1. Per-qid case count per size
    print("=" * 78)
    print("PER-QID SUB-LATTICE COUNT (rows = qid, cols = k)")
    print("=" * 78)
    qids = sorted(set(r["qid"] for r in results), key=lambda q: int(q))
    sizes = sorted(by_size)
    header = "  qid       L_full | " + " ".join(f"k={k:>2}" for k in sizes) + "  total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for qid in qids:
        row_counts = Counter(r["size"] for r in results if r["qid"] == qid)
        row = f"  {qid:<9} L{cf_layer_full[qid]:<6}|"
        for k in sizes:
            row += f"  {row_counts.get(k, 0):>3}"
        row += f"  {sum(row_counts.values()):>5}"
        print(row)

    # 2. Effective CF layer distribution per size
    print("\n" + "=" * 78)
    print("EFFECTIVE BFS CF-LAYER DISTRIBUTION PER SIZE")
    print("(rows = effective CF layer in the sub-lattice, cols = k)")
    print("=" * 78)
    all_layers = sorted(set(r["bfs"]["layer"] for r in results))
    header = "  cf_layer | " + " ".join(f"k={k:>2}" for k in sizes)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cl in all_layers:
        row = f"  L{cl:<7}|"
        for k in sizes:
            n = sum(1 for r in by_size[k] if r["bfs"]["layer"] == cl)
            row += f"  {n:>3}" if n else f"   {' ':>3}"
        print(row)

    # 3. Per-size headline numbers (sanity check vs earlier table)
    print("\n" + "=" * 78)
    print("PER-SIZE HEADLINE (already excludes the 3 qids)")
    print("=" * 78)
    print(f"  {'k':>3} {'n':>3} | {'BFS_layer (mean)':>17} | {'BFS$_avg':>9}")
    print("  " + "-" * 50)
    for k in sizes:
        b = by_size[k]
        mean_layer = sum(r["bfs"]["layer"] for r in b) / len(b)
        mean_bfs = sum(r["bfs"]["calls"] for r in b) / len(b)
        print(f"  {k:>3} {len(b):>3} | {mean_layer:>17.1f} | {mean_bfs:>9.1f}")


if __name__ == "__main__":
    main()
