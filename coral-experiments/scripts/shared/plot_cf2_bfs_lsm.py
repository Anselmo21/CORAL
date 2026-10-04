"""BFS vs SRA vs ERA vs LSM on the cf2 case pool (all four score/prob runs).

Reads rows from logged_results/<run>-cf2/<dataset>/results.json (written by
scripts.shared.eval_cf2_bfs_sra_era: BFS + native-combsum SRA + mra_search
softmax b=4/8/16 labeled "ERA", plus score/prob-fed "ERA (elicited)" columns,
per PRESENTATION_PLOTS.md) and LSM from
cc_results_budget_{4,8,16}.json in the same dir (written by the GPU CC step).
LSM lines are drawn only for budgets whose file exists and covers every case
— partial checkpoints from still-running CC jobs are skipped with a note.

Reuses the panel-drawing code from plot_qwen3_temp0_bfs_lsm (same 3-panel
accuracy / cost / hop-distance layout, colors, and cost conventions).

Outputs per run:
    presentation_plots/<run>-cf2/<dataset>/<dataset>.png
    presentation_plots/<run>-cf2/bfs_lsm_mra_combined.png

Usage:
    conda run -n coral python3 -m scripts.shared.plot_cf2_bfs_lsm [--runs <run> ...] [--datasets <ds> ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.shared import plot_qwen3_temp0_bfs_lsm as panels
from scripts.shared.plot_qwen3_scores_bfs_sra import DATASET_SUBDIRS, RUNS as SCORE_RUNS
from scripts.shared.plot_style import apply_plot_style

RUNS = {run: label for run, (_, label) in SCORE_RUNS.items()}
DATASETS = [(ds, nice) for ds, _, nice in DATASET_SUBDIRS]
DS_NAMES = [ds for ds, _ in DATASETS]


def _load(run: str, dataset: str) -> tuple[list[dict], dict[int, list]]:
    root = Path(f"logged_results/{run}-cf2/{dataset}")
    rows = json.load(open(root / "results.json"))
    cc: dict[int, list] = {}
    by_id = {r["case_id"]: r for r in rows}
    for budget, *_ in panels.LSM_METHODS:
        # TEMPORARY: show only LSM b=8 — b=4/16 hidden until their CC re-run
        # on the cfmin3 pool completes.
        if budget != 8:
            cc[budget] = []
            continue
        path = root / f"cc_results_budget_{budget}.json"
        data = json.load(open(path)) if path.exists() else []
        # Keep only rows whose recorded BFS matches the current pool's case —
        # drops partial checkpoints AND rows from a superseded pool (e.g. the
        # pre-cfmin2 exactly-2 pool, where only k=10 cases survive).
        kept = [c for c in data if c["case_id"] in by_id
                and c.get("bfs_key") == by_id[c["case_id"]]["bfs"]["key"]]
        # A budget is drawn in full only when its CC output covers every case
        # in the pool; otherwise fall back to the k=10 cases (old-pool k<=9
        # survivors are tiny coincidental samples — jumpy lines) until the CC
        # re-run on the cfmin3 pool lands for that budget.
        if len(kept) < len(rows):
            kept = [c for c in kept if by_id[c["case_id"]]["size"] == 10]
            print(f"    NOTE: {path}: partial coverage "
                  f"({len(kept)} k=10 rows kept of {len(data)} in file) — "
                  f"drawing this budget at k=10 only")
        cc[budget] = kept
    return rows, cc


def _legend(fig, handles: dict, y: float = 1.03) -> None:
    order = [panels.BFS_STYLE[0], panels.SRA_STYLE[1]] \
        + [t[2] for t in panels.ERA_METHODS] \
        + [lab for _, lab, *_ in panels.LSM_METHODS]
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, y), fontsize=14, frameon=True)


def _xlabel(dataset: str) -> str:
    return "Number of Available Pragmas" if "pragma" in dataset \
        else "Number of Available Criteria"


def _plot_per_dataset(run: str, dataset: str, plot_root: str) -> None:
    rows, cc = _load(run, dataset)
    out_dir = f"{plot_root}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    handles: dict[str, object] = {}
    panels._draw_panels(axes, rows, cc, _xlabel(dataset), handles)
    _legend(fig, handles)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{out_dir}/{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def _plot_combined(run: str, nice: str, plot_root: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    handles: dict[str, object] = {}
    for col, (dataset, ds_nice) in enumerate(DATASETS):
        rows, cc = _load(run, dataset)
        panels._draw_panels(axes[:, col], rows, cc, _xlabel(dataset), handles)
        axes[0, col].set_title(f"{nice} {ds_nice}")
    _legend(fig, handles, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{plot_root}/bfs_lsm_mra_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--datasets", nargs="+", default=DS_NAMES, choices=DS_NAMES)
    args = ap.parse_args()

    apply_plot_style()
    for run in args.runs:
        nice = RUNS[run]
        plot_root = f"presentation_plots/{run}-cf2"
        os.makedirs(plot_root, exist_ok=True)
        print(f"=== {nice} ===")
        for dataset, _ in DATASETS:
            if dataset in args.datasets:
                _plot_per_dataset(run, dataset, plot_root)
        if set(args.datasets) == set(DS_NAMES):
            _plot_combined(run, nice, plot_root)


if __name__ == "__main__":
    main()
