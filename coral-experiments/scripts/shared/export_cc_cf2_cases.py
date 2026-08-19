"""Export cc_cases.json for the cf2 case pool (CPU step, run before the GPU step).

Reads data/cases_16x20_cfmin3.json (20 sub-lattices per size k=3..9 per
(run, dataset), shallowest CF >= 3 removals below start for k>=4, >= 2 for k=3 — see
scripts.shared.generate_cf2_pool) plus the lattice runs under
coral-runs-lattices/<run>/, and writes the self-contained per-case exports
consumed by scripts.shared.run_cc_qwen:

    logged_results/<run>-cf2/<dataset>/cc_cases.json

Each case carries query/sources/plausible_answers from the run's config.json
and rankings_by_combo restricted to the sub-lattice, matching the schema of
the existing qwen3-32b-temp0 exports.

Usage:
    conda run -n coral python3 -m scripts.shared.export_cc_cf2_cases
"""
from __future__ import annotations

import json
import os
from itertools import combinations

from search.lattice import combo_key

CASES_PATH = "data/cases_16x20_cfmin3.json"
DATASET_SUBDIRS = {
    "msmarco_criteria": "msmarco-criteria",
    "msmarco_pragma":   "msmarco-pragmas",
    "curated_criteria": "curated-criteria",
    "curated_pragma":   "curated-pragmas",
}


def main() -> None:
    pool = json.load(open(CASES_PATH))
    for run, by_ds in pool["cases"].items():
        for ds, cases in by_ds.items():
            base = f"coral-runs-lattices/{run}/{DATASET_SUBDIRS[ds]}"
            cfg_cache: dict[str, tuple[dict, dict]] = {}
            out = []
            for i, case in enumerate(cases):
                qid = case["qid"]
                if qid not in cfg_cache:
                    cfg = json.load(open(f"{base}/{qid}/config.json"))
                    ans = json.load(open(f"{base}/{qid}/answers.json"))
                    pa = cfg["plausible_answers"]
                    rbc_full = {k: [pa[j] for j in e["ranking"]] for k, e in ans.items()}
                    cfg_cache[qid] = (cfg, rbc_full)
                cfg, rbc_full = cfg_cache[qid]
                start = case["start_node"]
                rbc = {}
                for r in range(len(start) + 1):
                    for combo in combinations(sorted(start), r):
                        key = combo_key(list(combo))
                        rbc[key] = rbc_full[key]
                out.append({
                    "case_id":           i,
                    "query_id":          qid,
                    "query":             cfg["query"],
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
