"""Evaluate MRA with softmax_entropy at different temperatures.

Runs mra_search with entropy_fn=softmax_entropy(T=0.1) and T=0.5,
plots against BFS, SRA, and the standard ERA (MRA softmax+shannon) baseline
for Llama-3.1-8B and Qwen3-32B across all 4 datasets.

Usage:
    conda run -n coral python3 -m scripts.shared.eval_mra_temperature
    conda run -n coral python3 -m scripts.shared.eval_mra_temperature --scores
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from search.lattice import combo_key, jaccard_keys
from search.rank import mra_search
from search.heuristics import softmax_entropy, shannon_entropy, score_gap_inv
from scripts.shared.plot_style import apply_plot_style, draw_hop_bars, bootstrap_ci_mean
from matplotlib.ticker import MaxNLocator

MODELS = {
    "llama31-8b": {
        "datasets": [
            ("msmarco_criteria", "data/llama31-8b/msmarco-criteria-lattices.json", "MSMARCO criteria"),
            ("msmarco_pragma",   "data/llama31-8b/msmarco-pragmas-lattices.json",  "MSMARCO pragma"),
            ("curated_criteria", "data/llama31-8b/curated-criteria-lattices.json",  "Curated criteria"),
            ("curated_pragma",   "data/llama31-8b/curated-pragmas-lattices.json",   "Curated pragma"),
        ],
        "result_root": "logged_results",
    },
    "qwen3-32b": {
        "datasets": [
            ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-lattices.json",  "MSMARCO criteria"),
            ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-lattices.json",   "MSMARCO pragma"),
            ("curated_criteria", "data/qwen3-32b/curated-criteria-lattices.json",   "Curated criteria"),
            ("curated_pragma",   "data/qwen3-32b/curated-pragmas-lattices.json",    "Curated pragma"),
        ],
        "result_root": "logged_results/qwen3-32b",
    },
}

BUDGETS = [4, 8, 16]

TEMPERATURES = [0.1, 0.5, 1.0]

BASELINE_LABEL = "ERA"  # MRA softmax+shannon (paper convention)

CORAL_RUNS_MAP = {
    "msmarco_criteria": "msmarco-criteria",
    "msmarco_pragma":   "msmarco-pragmas",
    "curated_criteria": "curated-criteria",
    "curated_pragma":   "curated-pragmas",
}


def _load_coral_runs_scores(ds_name: str, base: str = "coral-runs-lattices/qwen3-32b"):
    """Load scores_by_combo from coral-runs per-qid format.
    Returns {qid_str: scores_by_combo} where scores_by_combo maps
    combo_key -> {answer_string: float}.
    """
    subdir = CORAL_RUNS_MAP.get(ds_name)
    if subdir is None:
        return {}
    data_dir = Path(base) / subdir
    if not data_dir.exists():
        return {}
    out = {}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("halted"):
            continue
        ans = json.loads((d / "answers.json").read_text())
        pa = list(cfg["plausible_answers"])
        sbc = {}
        for key, entry in ans.items():
            sbc[key] = {pa[int(i)]: float(s) for i, s in entry.get("scores", {}).items()}
        out[d.name] = sbc
    return out


def _load_lattices(data_path: str) -> dict[str, tuple[list[str], dict]]:
    raw = json.load(open(data_path))
    out: dict[str, tuple[list[str], dict]] = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q.get("criteria") or [])
        rbc = q.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _score_proxy(bfs_result, proxy_result, rbc, start_node) -> dict:
    start_key = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    bfs_key = bfs_result.get("key")
    bfs_layer = bfs_result.get("size") or bfs_result.get("layer")
    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")
    proxy_verified = (proxy_key is not None
                      and bool(rbc.get(proxy_key))
                      and rbc[proxy_key][0] != baseline_top1)
    if not proxy_result.get("found") or proxy_key is None:
        outcome = "not_found"
    elif not proxy_verified:
        outcome = "false_positive"
    elif proxy_layer == bfs_layer:
        outcome = "correct"
    else:
        outcome = "wrong_layer"
    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": proxy_verified,
        "outcome": outcome,
        "jaccard": jaccard_keys(bfs_key, proxy_key),
        "calls": proxy_result.get("calls", 0),
        "budget_used": proxy_result.get("budget_used", 0),
        "pruned_nodes": proxy_result.get("pruned_nodes", 0),
    }


def evaluate_model(model: str, use_scores: bool = False):
    cfg = MODELS[model]
    result_root = cfg["result_root"]

    all_results = {}

    for ds_name, data_path, nice in cfg["datasets"]:
        print(f"\n  {model} / {nice}")
        lattices = _load_lattices(data_path)
        paper_path = f"{result_root}/{ds_name}/paper_results.json"
        rows = json.load(open(paper_path))
        rows = [r for r in rows if r["qid"] in lattices]

        scores_map = {}
        if use_scores:
            scores_map = _load_coral_runs_scores(ds_name)
            print(f"    loaded scores for {len(scores_map)} qids")

        for row in rows:
            qid = row["qid"]
            start = row["start_node"]
            if qid not in lattices:
                continue
            crit, rbc = lattices[qid]
            bfs_key = row["bfs"].get("key")
            bfs_layer = row["bfs"].get("layer")
            bfs_proxy = {"key": bfs_key, "size": bfs_layer, "found": row["bfs"]["found"]}

            sbc = scores_map.get(str(qid)) if use_scores else None

            for b in BUDGETS:
                col = f"era_b{b}"
                if col not in row or row[col] is None:
                    result = mra_search(start, rbc, budget=b, start_node=start,
                                        direction="down", normalize_fn="softmax",
                                        entropy_fn=shannon_entropy,
                                        scores_by_combo=sbc)
                    row[col] = _score_proxy(bfs_proxy, result, rbc, start)

                for T in TEMPERATURES:
                    col_t = f"mra_softmax_T{T}_b{b}"
                    entropy_fn = partial(softmax_entropy, T=T)
                    result = mra_search(start, rbc, budget=b, start_node=start,
                                        direction="down", normalize_fn="softmax",
                                        entropy_fn=entropy_fn,
                                        scores_by_combo=sbc)
                    row[col_t] = _score_proxy(bfs_proxy, result, rbc, start)

                col_sg = f"score_gap_b{b}"
                result = mra_search(start, rbc, budget=b, start_node=start,
                                    direction="down", normalize_fn="softmax",
                                    entropy_fn=score_gap_inv,
                                    scores_by_combo=sbc)
                row[col_sg] = _score_proxy(bfs_proxy, result, rbc, start)

        all_results[ds_name] = rows
        n = len(rows)
        for b in BUDGETS:
            era_acc = 100 * sum(1 for r in rows if r.get(f"era_b{b}", {}).get("outcome") == "correct") / n
            sg_acc = 100 * sum(1 for r in rows if r.get(f"score_gap_b{b}", {}).get("outcome") == "correct") / n
            for T in TEMPERATURES:
                col_t = f"mra_softmax_T{T}_b{b}"
                t_acc = 100 * sum(1 for r in rows if r.get(col_t, {}).get("outcome") == "correct") / n
                print(f"    b={b}: ERA={era_acc:.1f}%  T={T}={t_acc:.1f}%")
            print(f"    b={b}: ERA={era_acc:.1f}%  ScoreGap={sg_acc:.1f}%")

    return all_results


def _by_size(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return dict(sorted(out.items()))


def _series(rows_by, col, kind, is_bfs=False):
    xs, ys = [], []
    for size, bucket in rows_by.items():
        cells = [r.get(col) for r in bucket if r.get(col)]
        if not cells:
            continue
        if kind == "acc":
            if is_bfs:
                v = 100.0
            else:
                v = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)
        elif kind == "cost":
            if is_bfs:
                v = float(np.mean([c["calls"] for c in cells]))
            else:
                v = float(np.mean([c.get("calls", 0) + c.get("budget_used", 0) for c in cells]))
        else:
            v = 0.0
        xs.append(size)
        ys.append(v)
    return xs, ys


def _hop_by_size(rows, col, sizes):
    per_size = {s: [] for s in sizes}
    for r in rows:
        cell = r.get(col)
        if not cell or cell.get("outcome") == "correct":
            continue
        pl = cell.get("proxy_layer")
        bl = r["bfs"].get("layer")
        if pl is not None and bl is not None:
            per_size[r["size"]].append(float(abs(pl - bl)))
    return per_size


PLOT_METHODS_BASE = [
    ("bfs",  "BFS",  "o", "--", "black",   2.0, True),
    ("sra",  "SRA",  "s", "-",  "#d62728", 1.8, False),
]

ERA_COLORS = {4: "#1f77b4", 8: "#17becf", 16: "#9467bd"}
ERA_MARKERS = {4: "X", 8: "h", 16: "D"}

T01_COLORS = {4: "#2ca02c", 8: "#98df8a", 16: "#006400"}
T01_MARKERS = {4: "^", 8: "P", 16: "v"}

T05_COLORS = {4: "#ff7f0e", 8: "#ffbb78", 16: "#e377c2"}
T05_MARKERS = {4: ">", 8: "<", 16: "d"}

T10_COLORS = {4: "#8c564b", 8: "#c49c94", 16: "#bcbd22"}
T10_MARKERS = {4: "p", 8: "H", 16: "*"}

SG_COLORS = {4: "#7f7f7f", 8: "#aec7e8", 16: "#393b79"}
SG_MARKERS = {4: "2", 8: "1", 16: "3"}


def plot_model(model: str, all_results: dict, plot_suffix: str = ""):
    apply_plot_style()
    cfg = MODELS[model]
    plot_root = f"plots/mra_temperature/{model}{plot_suffix}"
    os.makedirs(plot_root, exist_ok=True)

    for ds_name, _, nice in cfg["datasets"]:
        rows = all_results[ds_name]
        rows_by = _by_size(rows)
        all_sizes = sorted(rows_by.keys())
        xlabel = "Number of Available Pragmas" if "pragma" in ds_name else "Number of Available Criteria"

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        handles = {}

        for ax, kind, ylabel in [(axes[0], "acc",  "Accuracy (%)"),
                                  (axes[1], "cost", "Avg Cost (LLM Calls)"),
                                  (axes[2], "hop",  "Distance to True Counterfactual")]:
            if kind == "hop":
                method_data = []
                method_data.append(("SRA", "#d62728",
                                    _hop_by_size(rows, "sra", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"{BASELINE_LABEL} (b={b})", ERA_COLORS[b],
                                        _hop_by_size(rows, f"era_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"T=0.1 (b={b})", T01_COLORS[b],
                                        _hop_by_size(rows, f"mra_softmax_T0.1_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"T=0.5 (b={b})", T05_COLORS[b],
                                        _hop_by_size(rows, f"mra_softmax_T0.5_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"T=1.0 (b={b})", T10_COLORS[b],
                                        _hop_by_size(rows, f"mra_softmax_T1.0_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"ScoreGap (b={b})", SG_COLORS[b],
                                        _hop_by_size(rows, f"score_gap_b{b}", all_sizes)))
                draw_hop_bars(ax, all_sizes, method_data, handles)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, alpha=0.3, axis="y")
            else:
                for col, lab, mk, ls, color, lw, is_bfs in PLOT_METHODS_BASE:
                    if is_bfs and kind == "acc":
                        continue
                    xs, ys = _series(rows_by, col, kind, is_bfs)
                    if xs:
                        line, = ax.plot(xs, ys, marker=mk, linestyle=ls, color=color,
                                        linewidth=lw, markersize=7, label=lab)
                        handles.setdefault(lab, line)

                for b in BUDGETS:
                    lab = f"{BASELINE_LABEL} (b={b})"
                    xs, ys = _series(rows_by, f"era_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=ERA_MARKERS[b], linestyle="-",
                                        color=ERA_COLORS[b], linewidth=1.8, markersize=7, label=lab)
                        handles.setdefault(lab, line)

                for b in BUDGETS:
                    lab = f"T=0.1 (b={b})"
                    xs, ys = _series(rows_by, f"mra_softmax_T0.1_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=T01_MARKERS[b], linestyle="-.",
                                        color=T01_COLORS[b], linewidth=1.8, markersize=7, label=lab)
                        handles.setdefault(lab, line)

                for b in BUDGETS:
                    lab = f"T=0.5 (b={b})"
                    xs, ys = _series(rows_by, f"mra_softmax_T0.5_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=T05_MARKERS[b], linestyle=":",
                                        color=T05_COLORS[b], linewidth=1.8, markersize=7, label=lab)
                        handles.setdefault(lab, line)

                for b in BUDGETS:
                    lab = f"T=1.0 (b={b})"
                    xs, ys = _series(rows_by, f"mra_softmax_T1.0_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=T10_MARKERS[b], linestyle=(0, (3, 1, 1, 1)),
                                        color=T10_COLORS[b], linewidth=1.8, markersize=7, label=lab)
                        handles.setdefault(lab, line)

                for b in BUDGETS:
                    lab = f"ScoreGap (b={b})"
                    xs, ys = _series(rows_by, f"score_gap_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=SG_MARKERS[b], linestyle=(0, (5, 2)),
                                        color=SG_COLORS[b], linewidth=1.8, markersize=8, label=lab)
                        handles.setdefault(lab, line)

                ax.grid(True, alpha=0.3)
                if kind == "acc":
                    ax.set_ylim(0, 108)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(all_sizes)

        order = (["BFS", "SRA"]
                 + [f"{BASELINE_LABEL} (b={b})" for b in BUDGETS]
                 + [f"T=0.1 (b={b})" for b in BUDGETS]
                 + [f"T=0.5 (b={b})" for b in BUDGETS]
                 + [f"T=1.0 (b={b})" for b in BUDGETS]
                 + [f"ScoreGap (b={b})" for b in BUDGETS])
        ordered = [l for l in order if l in handles]
        fig.legend([handles[l] for l in ordered], ordered,
                   loc="upper center", ncol=min(len(ordered), 6),
                   bbox_to_anchor=(0.5, 1.05), fontsize=9, frameon=True)
        model_label = "Llama-3.1-8B" if "llama" in model else "Qwen3-32B"
        score_tag = " (Model Scores)" if plot_suffix else ""
        fig.suptitle(f"{model_label} — {nice}{score_tag}", fontsize=14, y=1.12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_dir = f"{plot_root}/{ds_name}"
        os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/temperature_ablation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    _plot_combined(model, all_results, plot_root, cfg, plot_suffix)


def _plot_combined(model: str, all_results: dict, plot_root: str, cfg: dict,
                   plot_suffix: str = ""):
    datasets = cfg["datasets"]
    fig, axes = plt.subplots(3, len(datasets), figsize=(5 * len(datasets), 12))
    handles = {}

    for col_idx, (ds_name, _, nice) in enumerate(datasets):
        rows = all_results[ds_name]
        rows_by = _by_size(rows)
        all_sizes = sorted(rows_by.keys())
        xlabel = "Number of Available Pragmas" if "pragma" in ds_name else "Number of Available Criteria"

        for row_idx, (kind, ylabel) in enumerate([("acc", "Accuracy (%)"),
                                                    ("cost", "Avg Cost (LLM Calls)"),
                                                    ("hop", "Distance to True CF")]):
            ax = axes[row_idx, col_idx]
            if kind == "hop":
                method_data = []
                method_data.append(("SRA", "#d62728",
                                    _hop_by_size(rows, "sra", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"{BASELINE_LABEL} (b={b})", ERA_COLORS[b],
                                        _hop_by_size(rows, f"era_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"T=0.1 (b={b})", T01_COLORS[b],
                                        _hop_by_size(rows, f"mra_softmax_T0.1_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"T=0.5 (b={b})", T05_COLORS[b],
                                        _hop_by_size(rows, f"mra_softmax_T0.5_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"T=1.0 (b={b})", T10_COLORS[b],
                                        _hop_by_size(rows, f"mra_softmax_T1.0_b{b}", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"ScoreGap (b={b})", SG_COLORS[b],
                                        _hop_by_size(rows, f"score_gap_b{b}", all_sizes)))
                draw_hop_bars(ax, all_sizes, method_data, handles)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, alpha=0.3, axis="y")
            else:
                for c, lab, mk, ls, color, lw, is_bfs in PLOT_METHODS_BASE:
                    if is_bfs and kind == "acc":
                        continue
                    xs, ys = _series(rows_by, c, kind, is_bfs)
                    if xs:
                        line, = ax.plot(xs, ys, marker=mk, linestyle=ls, color=color,
                                        linewidth=lw, markersize=6, label=lab)
                        handles.setdefault(lab, line)
                for b in BUDGETS:
                    lab = f"{BASELINE_LABEL} (b={b})"
                    xs, ys = _series(rows_by, f"era_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=ERA_MARKERS[b], linestyle="-",
                                        color=ERA_COLORS[b], linewidth=1.5, markersize=6, label=lab)
                        handles.setdefault(lab, line)
                for b in BUDGETS:
                    lab = f"T=0.1 (b={b})"
                    xs, ys = _series(rows_by, f"mra_softmax_T0.1_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=T01_MARKERS[b], linestyle="-.",
                                        color=T01_COLORS[b], linewidth=1.5, markersize=6, label=lab)
                        handles.setdefault(lab, line)
                for b in BUDGETS:
                    lab = f"T=0.5 (b={b})"
                    xs, ys = _series(rows_by, f"mra_softmax_T0.5_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=T05_MARKERS[b], linestyle=":",
                                        color=T05_COLORS[b], linewidth=1.5, markersize=6, label=lab)
                        handles.setdefault(lab, line)
                for b in BUDGETS:
                    lab = f"T=1.0 (b={b})"
                    xs, ys = _series(rows_by, f"mra_softmax_T1.0_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=T10_MARKERS[b], linestyle=(0, (3, 1, 1, 1)),
                                        color=T10_COLORS[b], linewidth=1.5, markersize=6, label=lab)
                        handles.setdefault(lab, line)
                for b in BUDGETS:
                    lab = f"ScoreGap (b={b})"
                    xs, ys = _series(rows_by, f"score_gap_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=SG_MARKERS[b], linestyle=(0, (5, 2)),
                                        color=SG_COLORS[b], linewidth=1.5, markersize=7, label=lab)
                        handles.setdefault(lab, line)
                ax.grid(True, alpha=0.3)
                if kind == "acc":
                    ax.set_ylim(0, 108)

            ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(all_sizes)
            if row_idx == 0:
                ax.set_title(nice)

    order = (["BFS", "SRA"]
             + [f"{BASELINE_LABEL} (b={b})" for b in BUDGETS]
             + [f"T=0.1 (b={b})" for b in BUDGETS]
             + [f"T=0.5 (b={b})" for b in BUDGETS]
             + [f"T=1.0 (b={b})" for b in BUDGETS]
             + [f"ScoreGap (b={b})" for b in BUDGETS])
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=min(len(ordered), 6),
               bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)
    model_label = "Llama-3.1-8B" if "llama" in model else "Qwen3-32B"
    score_tag = " (Model Scores)" if plot_suffix else ""
    fig.suptitle(f"{model_label} — MRA Softmax Entropy Temperature Ablation{score_tag}", fontsize=15, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{plot_root}/combined_temperature_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", action="store_true",
                        help="Use coral-runs model scores instead of Borda (qwen3-32b only)")
    args = parser.parse_args()

    if args.scores:
        model = "qwen3-32b"
        print(f"\n{'='*60}\n  {model} (model scores)\n{'='*60}")
        results = evaluate_model(model, use_scores=True)
        plot_model(model, results, plot_suffix="-scores")
    else:
        for model in ["llama31-8b", "qwen3-32b"]:
            print(f"\n{'='*60}\n  {model}\n{'='*60}")
            results = evaluate_model(model)
            plot_model(model, results)


if __name__ == "__main__":
    main()
