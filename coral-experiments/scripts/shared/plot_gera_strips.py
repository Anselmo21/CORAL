"""Qwen3-32B result strips: every method at one budget.

Metric per strip, dataset per column, every strip self-contained: its own
column titles, x-axes and per-panel x-labels, margins pinned in inches so
the strips stack flush at \\textwidth.

Series: BFS, SRA, ERA, GRA, GERA and LSM, all at one budget; the budget
sweep figure carries the same arms across budgets. The ERA/GRA/GERA cells
come from results/gera_suite_cells/ (scripts.graph_cut.run_gera_suite); SRA
and BFS ride along in the same dumps; LSM rows are read from the logged
cc_results_budget_<b>.json and filtered to the suite's covered cases.

Writes results/paper_figs/gera/rows/{legend,accuracy,cost,distance}.png.

Usage:
    python3 -m scripts.shared.plot_gera_strips --budget 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.shared.plot_qwen3_temp0_bfs_lsm as P
from scripts.shared.plot_style import apply_plot_style

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "gera_suite_cells"
LOGGED = REPO / "logged_results"
OUT_DIR = REPO / "results" / "paper_figs" / "gera" / "rows"

MODEL = "qwen3-32b"
DATASETS = ["msmarco_criteria", "msmarco_pragma", "curated_criteria", "curated_pragma"]
COLUMN_TITLES = {
    "msmarco_criteria": "MSMARCO criteria",
    "msmarco_pragma":   "MSMARCO pragma",
    "curated_criteria": "Curated criteria",
    "curated_pragma":   "Curated pragma",
}
METRICS = [(0, "accuracy"), (1, "cost"), (2, "distance")]

MARGIN_LEFT = 1.05
MARGIN_RIGHT = 0.12
PANEL_HEIGHT = 2.80
TOP_TITLE = 0.46
TOP_BARE = 0.10
BOTTOM_XAXIS = 0.86
BOTTOM_BARE = 0.10

PANEL_CENTER = 0.5


def load_rows(ds: str, budget: int):
    rows = json.loads((CELLS / f"{MODEL}_{ds}.json").read_text(encoding="utf-8"))
    for r in rows:
        r["bfs"]["layer"] = r["bfs"]["size"]
    covered = {(r["qid"], tuple(r["start_node"])) for r in rows}
    cc = [r for r in json.loads(
        (LOGGED / MODEL / ds / f"cc_results_budget_{budget}.json").read_text(encoding="utf-8"))
        if (str(r["query_id"]), tuple(r["start_node"])) in covered]
    return rows, {budget: cc}


def draw_metric(axes, slot: int, budget: int, handles: dict) -> None:
    """One 4-wide strip of a single metric; the unwanted panels go to scratch."""
    for col, ds in enumerate(DATASETS):
        rows, cc = load_rows(ds, budget)
        xlabel = ("Number of Available Pragmas" if "pragma" in ds
                  else "Number of Available Criteria")
        scratch, spare = plt.subplots(1, 3)
        triple = list(spare)
        triple[slot] = axes[col]
        P._draw_panels(triple, rows, cc, xlabel, handles)
        plt.close(scratch)
        axes[col].set_title(COLUMN_TITLES[ds], fontsize=15, pad=8)
        if col > 0:
            axes[col].set_ylabel("")
        elif slot == 2:
            # the one-line label overruns the strip; the caption names it anyway
            axes[col].set_ylabel("Distance to\nTrue Counterfactual")


def plot_legend_strip(handles: dict) -> Path:
    order = ([P.BFS_STYLE[0], P.SRA_STYLE[1]]
             + [t[2] for t in P.ERA_METHODS]
             + [lab for _, lab, *_ in P.LSM_METHODS])
    ordered = [l for l in order if l in handles]
    fig = plt.figure(figsize=(16, 0.58))
    fig.legend([handles[l] for l in ordered], ordered,
               loc="center", bbox_to_anchor=(PANEL_CENTER, 0.5),
               ncol=len(ordered), fontsize=14, frameon=True)
    out = OUT_DIR / "legend.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(budget: int, width: float) -> None:
    global PANEL_CENTER
    # existing series keep the figure's own colours; ERA and GRA wear the
    # purple and orange the old budget variants wore
    P.ERA_METHODS = [(budget, f"era_b{budget}", "ERA", "h", "#9467bd", "mra"),
                     (budget, f"gra_b{budget}", "GRA", "D", "#ff7f0e", "mra"),
                     (budget, f"gera_b{budget}", "GERA", "X", "#1f77b4", "mra")]
    P.LSM_METHODS = [(budget, "LSM", "^", "#2ca02c")]

    apply_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_handles: dict[str, object] = {}

    for slot, metric in METRICS:
        top = TOP_TITLE
        bottom = BOTTOM_XAXIS
        figure_height = top + PANEL_HEIGHT + bottom
        fig, axes = plt.subplots(1, 4, figsize=(width, figure_height))
        handles: dict[str, object] = {}
        draw_metric(axes, slot, budget, handles)
        all_handles.update(handles)

        fig.subplots_adjust(left=MARGIN_LEFT / width, right=1 - MARGIN_RIGHT / width,
                            top=1 - top / figure_height, bottom=bottom / figure_height,
                            wspace=0.22)
        PANEL_CENTER = (MARGIN_LEFT / width + (1 - MARGIN_RIGHT / width)) / 2
        out = OUT_DIR / f"{metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {out}")

    print(f"  -> {plot_legend_strip(all_handles)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--width", type=float, default=16.0)
    args = ap.parse_args()
    main(args.budget, args.width)
