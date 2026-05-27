import json
from itertools import combinations
from collections import Counter
from search.lattice import combo_key


def extract_sublattices(
    rows: list[dict],
    min_k: int = 3,
    max_k: int | None = None,
    min_dominance: float = 0.5,
) -> list[dict]:
    """Extract sub-lattices from larger lattices in the dataset.

    For each source lattice and each k-sized subset of its criteria, build
    the 2^k sub-lattice from the existing rankings_by_combo data.  Keep only
    sub-lattices where:
      1. A counterfactual exists (some node has a different top-1 than the
         start node).
      2. The start node's top-1 answer is the dominant answer (appears as
         top-1 in >= min_dominance fraction of the sub-lattice nodes).
    """
    results: list[dict] = []

    for row in rows:
        for run in row["baseline_lattice_runs"]:
            all_criteria = run["criteria_used"]
            rbc = run["rankings_by_combo"]
            n = len(all_criteria)
            upper = max_k if max_k is not None else n

            for k in range(min_k, min(upper, n) + 1):
                for combo in combinations(all_criteria, k):
                    subset_criteria = list(combo)
                    start_key = combo_key(subset_criteria)

                    start_ranking = rbc.get(start_key)
                    if not start_ranking:
                        continue
                    baseline_top1 = start_ranking[0]

                    sub_rbc: dict[str, list[str]] = {}
                    for sub_k in range(0, k + 1):
                        for sub_combo in combinations(subset_criteria, sub_k):
                            key = combo_key(sub_combo)
                            if key in rbc:
                                sub_rbc[key] = rbc[key]

                    top1_counts = Counter(
                        r[0] for r in sub_rbc.values() if r
                    )
                    total_nodes = len(sub_rbc)
                    dominance = top1_counts.get(baseline_top1, 0) / total_nodes

                    if dominance < min_dominance:
                        continue

                    has_cf = any(
                        r[0] != baseline_top1
                        for key, r in sub_rbc.items()
                        if key != start_key and r
                    )
                    if not has_cf:
                        continue

                    results.append({
                        "source_query_id": row["query_id"],
                        "criteria": subset_criteria,
                        "rankings_by_combo": sub_rbc,
                        "start_node": subset_criteria,
                        "baseline_top1": baseline_top1,
                        "dominance": round(dominance, 3),
                        "num_nodes": total_nodes,
                    })

    return results


if __name__ == "__main__":
    with open("data/base.json") as f:
        data = json.load(f)

    sublattices = extract_sublattices(data["rows"], min_k=3, max_k=6)

    print(f"Extracted {len(sublattices)} sub-lattices\n")

    by_size = Counter(s["num_nodes"] for s in sublattices)
    for size in sorted(by_size):
        print(f"  {size:>5} nodes (k={size.bit_length()-1}): {by_size[size]} sub-lattices")

    print(f"\nDominance stats:")
    dominances = [s["dominance"] for s in sublattices]
    print(f"  min={min(dominances):.3f}  max={max(dominances):.3f}  "
          f"mean={sum(dominances)/len(dominances):.3f}")

    with open("data/sublattices.json", "w") as f:
        json.dump(sublattices, f, indent=2)
    print(f"\nSaved to data/sublattices.json")
