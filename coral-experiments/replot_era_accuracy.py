"""One-off replot for data/era_ablation_study/dev/accuracy_by_size_down.png.

Reads era_budget_ablation_results.json and regenerates the accuracy-by-size
plot with a wider canvas and a legend that fits.
"""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

BUDGETS: list[int] = [4, 8, 16, 32, 64, 128, 256]
RESULTS_PATH = "data/era_ablation_study/dev/era_budget_ablation_results.json"
OUT_DIR = "data/era_ablation_study/dev"


def replot(direction: str) -> None:
    apply_plot_style()

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    dir_results = [r for r in results if r["direction"] == direction]
    if not dir_results:
        print(f"No results for direction={direction}")
        return

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in dir_results:
        by_size[r["effective_size"]].append(r)
    sizes = sorted(by_size)

    cmap = plt.cm.viridis
    n_b = max(1, len(BUDGETS) - 1)
    budget_colors = {b: cmap(i / n_b) for i, b in enumerate(BUDGETS)}
    markers = ["o", "v", "s", "^", "D", "P", "X"]
    budget_markers = {b: markers[i % len(markers)] for i, b in enumerate(BUDGETS)}
    dir_label = {"down": "Top-Down", "up": "Bottom-Up"}[direction]

    fig, ax = plt.subplots(figsize=(11, 6))
    for budget in BUDGETS:
        bkey = f"budget_{budget}"
        accs = [
            sum(1 for r in by_size[s] if r[bkey]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]
        ax.plot(
            sizes,
            accs,
            f"{budget_markers[budget]}-",
            label=f"budget={budget}",
            color=budget_colors[budget],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Number of Criteria")
    ax.set_ylabel("Accuracy")
    ax.set_title(dir_label)
    ax.set_xticks(sizes)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        frameon=True,
    )

    fig.suptitle("ERA Accuracy by Budget Size", fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    os.makedirs(OUT_DIR, exist_ok=True)
    notitle_dir = os.path.join(OUT_DIR, "notitle")
    os.makedirs(notitle_dir, exist_ok=True)

    filename = f"accuracy_by_size_{direction}.png"
    titled = os.path.join(OUT_DIR, filename)
    fig.savefig(titled, dpi=150, bbox_inches="tight")
    print(f"Saved {titled}")

    fig.suptitle("")
    notitle = os.path.join(notitle_dir, filename)
    fig.savefig(notitle, dpi=150, bbox_inches="tight")
    print(f"Saved {notitle}")

    plt.close(fig)


if __name__ == "__main__":
    replot("down")
