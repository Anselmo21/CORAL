"""Cross-dataset ERA accuracy vs budget plot.

Reads each of the four datasets' `results.json` files and renders a two-panel
figure: fixed budgets (log-x) on the left, pct budgets (linear-x) on the right.
SRA accuracy is plotted as a horizontal dashed line per dataset.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.shared.plot_style import apply_plot_style


DATASETS: dict[str, tuple[str, str]] = {
    "msmarco":        ("data/msmarco_criteria/results.json", "dict"),
    "sonnet":         ("data/sonnet_criteria/results.json",  "dict"),
    "msmarco_pragma": ("data/msmarco_pragma/results.json",    "list"),
    "sonnet_pragma":  ("data/sonnet_pragma/results.json",     "list"),
}

FIXED_BUDGETS: list[int] = [4, 8, 16, 32, 64, 128, 256]
PCT_BUDGETS: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

OUT_PATH = Path("plots/era_budget_vs_accuracy.png")


def load_rows(path: str, kind: str) -> list[dict]:
    d = json.load(open(path))
    return d["results"] if kind == "dict" else d


def accuracy(rows: list[dict], budget_key: str) -> float | None:
    n = sum(1 for r in rows if budget_key in r["era"])
    c = sum(1 for r in rows if budget_key in r["era"] and r["era"][budget_key].get("outcome") == "correct")
    return c / n if n else None


def sra_accuracy(rows: list[dict]) -> float:
    return sum(1 for r in rows if r["sra"].get("outcome") == "correct") / len(rows)


def main() -> None:
    apply_plot_style()
    fig, (ax_fixed, ax_pct) = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.get_cmap("tab10").colors

    for i, (name, (path, kind)) in enumerate(DATASETS.items()):
        rows = load_rows(path, kind)
        color = colors[i]

        fixed_acc = [accuracy(rows, f"era_fixed_{b}") for b in FIXED_BUDGETS]
        pct_acc = [accuracy(rows, f"era_pct_{p}") for p in PCT_BUDGETS]
        sra = sra_accuracy(rows) * 100

        ax_fixed.plot(FIXED_BUDGETS, [a * 100 for a in fixed_acc],
                      marker="o", color=color, label=f"{name} (n={len(rows)})")
        ax_fixed.axhline(sra, color=color, linestyle="--", linewidth=1, alpha=0.5)

        ax_pct.plot(PCT_BUDGETS, [a * 100 for a in pct_acc],
                    marker="o", color=color, label=f"{name} (n={len(rows)})")
        ax_pct.axhline(sra, color=color, linestyle="--", linewidth=1, alpha=0.5)

    ax_fixed.set_xscale("log", base=2)
    ax_fixed.set_xticks(FIXED_BUDGETS)
    ax_fixed.set_xticklabels(FIXED_BUDGETS)
    ax_fixed.set_xlabel("Fixed budget")
    ax_fixed.set_ylabel("Accuracy (%)")
    ax_fixed.set_title("ERA accuracy vs fixed budget")
    ax_fixed.grid(True, alpha=0.3)
    ax_fixed.legend(loc="lower right")

    ax_pct.set_xticks(PCT_BUDGETS)
    ax_pct.set_xlabel("Percent budget (% of $2^k$)")
    ax_pct.set_ylabel("Accuracy (%)")
    ax_pct.set_title("ERA accuracy vs pct budget")
    ax_pct.grid(True, alpha=0.3)
    ax_pct.legend(loc="lower right")

    fig.suptitle("ERA budget → accuracy across datasets (dashed = SRA baseline)")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
