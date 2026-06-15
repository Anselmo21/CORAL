"""ERA budget ablation on the Sonnet pragma lattice dev set.

Loads the 8 dev cases from data/sonnet-pragma-dataset.json (split from
data/sonnet_pragma/split.json) and sweeps the same budget grid as the other
ERA ablation studies:
  Fixed:   [4, 8, 16, 32, 64, 128, 256]
  Percent: [10, 20, ..., 100]% of 2^10

Writes data/sonnet_pragma/era_ablation_results.json.

CLI: conda run -n coral python3 run_era_sonnet_pragma_ablation.py
"""
from __future__ import annotations

import json
import os

from era_budget_ablation import (
    BUDGETS,
    BUDGET_PCTS,
    print_summary,
    run_era_budget_ablation,
)
from plot_style import apply_plot_style
from run_cc_sonnet_pragma_ablation import DATASET_PATH, SPLIT_PATH, load_cases

OUT_PATH = "data/sonnet_pragma/era_ablation_results.json"


def main() -> None:
    apply_plot_style()

    cases = load_cases(DATASET_PATH, SPLIT_PATH)
    print(f"Loaded {len(cases)} dev cases")

    for case in cases:
        case["direction"] = "down"

    print(f"\nRunning ERA — fixed budgets {BUDGETS} ...")
    results_fixed = run_era_budget_ablation(cases, mode="absolute")
    print_summary(results_fixed, "down", mode="absolute")

    print(f"\nRunning ERA — pct budgets {BUDGET_PCTS}% ...")
    results_pct = run_era_budget_ablation(cases, mode="percent")
    print_summary(results_pct, "down", mode="percent")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"fixed": results_fixed, "pct": results_pct}, f, indent=2)
    print(f"\nResults → {OUT_PATH}")


if __name__ == "__main__":
    main()
