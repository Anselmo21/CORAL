"""Export cc_cases.json for a cf2/ranked case pool (CPU step, run before the GPU step).

Reads a case pool (default data/cases_16x20_cfmin3.json; see
scripts.shared.generate_cf2_pool / generate_ranked_pool) plus the source
lattices, and writes the self-contained per-case exports consumed by
scripts.shared.run_cc_qwen:

    logged_results/<run>-cf2/<dataset>/cc_cases.json

Each case carries query/sources/plausible_answers from the run's config.json
and rankings_by_combo restricted to the sub-lattice.

Two source formats are auto-detected per run:
  - score-elicited runs (coral-runs-lattices/<run>/<subdir>): config has
    query/atoms, answers.json maps combo_key -> {ranking: [idx], scores};
  - ranked runs (<run> = "<model>-ranked", from --ranked-lattice-root/
    <subdir>-<model>): config has question/criteria, answers.json maps
    combo_key -> [answer str, ...] directly (sonnet-* dir naming for curated).

Usage:
    conda run -n coral python3 -m scripts.shared.export_cc_cf2_cases \
        [--cases-path data/cases_16x20_cfmin2_qwen35-122b-ranked.json]
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

from search.lattice import combo_key

DATASET_SUBDIRS = {
    "msmarco_criteria": "msmarco-criteria",
    "msmarco_pragma":   "msmarco-pragmas",
    "curated_criteria": "curated-criteria",
    "curated_pragma":   "curated-pragmas",
}
RANKED_SUBDIRS = {**DATASET_SUBDIRS,
                  "curated_criteria": "sonnet-criteria",
                  "curated_pragma":   "sonnet-pragmas"}


def _load(base: str, qid: str) -> tuple[str, dict, dict]:
    """Returns (query, config, rankings_by_combo) for either source format."""
    cfg = json.load(open(f"{base}/{qid}/config.json"))
    ans = json.load(open(f"{base}/{qid}/answers.json"))
    if "atoms" in cfg:   # score-elicited coral-runs format
        pa = cfg["plausible_answers"]
        return cfg["query"], cfg, {k: [pa[j] for j in e["ranking"]] for k, e in ans.items()}
    return cfg["question"], cfg, ans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-path", default="data/cases_16x20_cfmin3.json")
    ap.add_argument("--ranked-lattice-root", default="/Users/kathling/coral-data-gen/lattices")
    args = ap.parse_args()

    pool = json.load(open(args.cases_path))
    for run, by_ds in pool["cases"].items():
        ranked = run.endswith("-ranked")
        for ds, cases in by_ds.items():
            if ranked:
                model = run[: -len("-ranked")]
                base = f"{args.ranked_lattice_root}/{RANKED_SUBDIRS[ds]}-{model}"
            else:
                base = f"coral-runs-lattices/{run}/{DATASET_SUBDIRS[ds]}"
            cache: dict[str, tuple[str, dict, dict]] = {}
            out = []
            for i, case in enumerate(cases):
                qid = case["qid"]
                if qid not in cache:
                    cache[qid] = _load(base, qid)
                query, cfg, rbc_full = cache[qid]
                start = case["start_node"]
                rbc = {}
                for r in range(len(start) + 1):
                    for combo in combinations(sorted(start), r):
                        key = combo_key(list(combo))
                        rbc[key] = rbc_full[key]
                out.append({
                    "case_id":           i,
                    "query_id":          qid,
                    "query":             query,
                    "sources":           cfg["sources"],
                    "plausible_answers": cfg["plausible_answers"],
                    "criteria":          start,
                    "start_node":        start,
                    "direction":         "down",
                    "effective_size":    case["size"],
                    "k":                 case["size"],
                    "cf_removals":       case["cf_removals"],
                    "rankings_by_combo": rbc,
                })
            out_dir = f"logged_results/{run}-cf2/{ds}"
            os.makedirs(out_dir, exist_ok=True)
            path = f"{out_dir}/cc_cases.json"
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"  {path}: {len(out)} cases")


if __name__ == "__main__":
    main()
