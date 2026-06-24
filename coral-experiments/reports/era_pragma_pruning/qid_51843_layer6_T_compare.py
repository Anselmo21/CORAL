"""Compare layer-6 entropy distribution for qid 51843 at T=1 vs T=0.5,
for both ERA(pct_10) and ERA(b=16). Annotates which side is kept vs pruned.
"""
from __future__ import annotations

import os
from functools import partial
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.pragma.msmarco_regen.run_subsample import _load_lattices
from scripts.shared.plot_style import apply_plot_style
from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key, iter_search_layers
from search.rank import _aggregate_with_scores, _get_child_rankings

OUT_DIR = "reports/era_pragma_pruning/figures"


def reach_layer6(criteria, rbc, start_node, budget, T):
    aggregator = "combsum"
    entropy_fn = partial(softmax_entropy, T=T)
    cutoff_fn = max_jump_cutoff
    start_key = combo_key(start_node)
    baseline = rbc[start_key]
    baseline_answer = baseline[0]
    answers = list(baseline)
    s = len(start_node)

    known: dict[str, list[str]] = {}
    estimated: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}

    for c in criteria:
        known[c] = rbc[c]

    budget_rem = budget
    for layer_size in range(2, s):
        layer_nodes = []
        for combo in combinations(range(s), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)
            voters = _get_child_rankings(subset, known, estimated)
            if not voters:
                continue
            ranking, scores = _aggregate_with_scores(voters, answers, aggregator)
            if not scores:
                continue
            estimated[key] = ranking
            entropies[key] = entropy_fn(scores)
            layer_nodes.append((key, entropies[key]))
        selected = cutoff_fn(layer_nodes, budget_rem)
        for key in selected:
            known[key] = rbc[key]
            budget_rem -= 1

    # Phase 2 — walk down to layer 6
    for layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        if layer_size > 6:
            unknown_nodes = []
            for subset in subsets:
                key = combo_key(subset)
                if key in known:
                    if known[key] and known[key][0] != baseline_answer:
                        return None
                    continue
                if key not in estimated:
                    unknown_nodes.append((subset, key, 0, float("inf")))
                    continue
                ranking = estimated[key]
                ent = entropies.get(key, 0.0)
                disp = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
                unknown_nodes.append((subset, key, disp, ent))
            if not unknown_nodes:
                continue
            entropy_inputs = [(k, e) for _, k, _, e in unknown_nodes]
            high_ent = set(cutoff_fn(entropy_inputs, len(unknown_nodes)))
            candidates = [(s_, k_, d_, e_) for s_, k_, d_, e_ in unknown_nodes
                          if not (d_ < 1 and k_ not in high_ent)]
            candidates.sort(key=lambda x: (-x[2], x[3]))
            for subset, key, _, _ in candidates:
                gt = rbc.get(key)
                if gt:
                    known[key] = gt
                if gt and gt[0] != baseline_answer:
                    return None
        elif layer_size == 6:
            unknown_nodes = []
            cf_keys: set[str] = set()
            for subset in subsets:
                key = combo_key(subset)
                if key in known:
                    if known[key] and known[key][0] != baseline_answer:
                        cf_keys.add(key)
                    continue
                if key not in estimated:
                    unknown_nodes.append((subset, key, 0, float("inf")))
                    continue
                ranking = estimated[key]
                ent = entropies.get(key, 0.0)
                disp = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
                unknown_nodes.append((subset, key, disp, ent))
                gt = rbc.get(key)
                if gt and gt[0] != baseline_answer:
                    cf_keys.add(key)
            return {"unknown_nodes": unknown_nodes, "cf_keys": cf_keys,
                    "n_known": sum(1 for s_ in subsets if combo_key(s_) in known),
                    "total": len(subsets)}
    return None


def plot_panel(ax, state, title, color):
    """Sort desc by entropy. Left side = KEPT (high entropy). Right side = PRUNED (low entropy)."""
    nodes = state["unknown_nodes"]
    cf_keys = state["cf_keys"]
    sorted_nodes = sorted(nodes, key=lambda n: n[3], reverse=True)
    entropies = [n[3] for n in sorted_nodes]
    keys = [n[1] for n in sorted_nodes]
    is_cf = [k in cf_keys for k in keys]

    gaps = [entropies[i] - entropies[i+1] for i in range(len(entropies) - 1)]
    if not gaps:
        return
    max_gap_idx = max(range(len(gaps)), key=lambda i: (gaps[i], i))
    cutoff_pos = max_gap_idx + 1   # nodes 0..max_gap_idx kept, the rest pruned

    n_kept = cutoff_pos
    n_pruned = len(entropies) - cutoff_pos
    n_cf_kept   = sum(is_cf[:cutoff_pos])
    n_cf_pruned = sum(is_cf[cutoff_pos:])

    x = np.arange(len(entropies))

    # Shaded background bands
    ax.axvspan(-0.5, cutoff_pos - 0.5, color="#2ca02c", alpha=0.08, zorder=0)
    ax.axvspan(cutoff_pos - 0.5, len(entropies) - 0.5, color="#d62728", alpha=0.08, zorder=0)

    # Bars: green if a CF, gray otherwise (so CFs pop)
    bar_colors = ["#2ca02c" if c else color for c in is_cf]
    ax.bar(x, entropies, color=bar_colors, width=1.0, alpha=0.9, edgecolor="none", zorder=2)

    # Cutoff line
    ax.axvline(cutoff_pos - 0.5, color="red", linestyle="--", linewidth=1.7, zorder=3,
               label=f"cutoff at idx {cutoff_pos}")

    # Annotations
    ax.text(cutoff_pos / 2 - 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else max(entropies) * 0.95,
            f"← KEPT ({n_kept} nodes, {n_cf_kept} CFs)",
            ha="center", va="top", fontsize=9, color="#2ca02c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2ca02c", alpha=0.85))
    pruned_center = cutoff_pos + n_pruned / 2 - 0.5
    ax.text(pruned_center, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else max(entropies) * 0.95,
            f"PRUNED ({n_pruned} nodes, {n_cf_pruned} CFs) →",
            ha="center", va="top", fontsize=9, color="#d62728", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.85))

    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_ylabel("entropy (bits)")
    ax.set_xlabel("nodes sorted by entropy (desc)")
    ax.grid(True, alpha=0.3, axis="y", zorder=1)


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    qid = "51843"
    lats = _load_lattices([qid])
    criteria, rbc = lats[qid]
    start_node = list(criteria)

    states = {}
    for T in [1.0, 0.5]:
        states[("pct_10", T)] = reach_layer6(criteria, rbc, start_node,
                                              budget=max(1, int(2**10 * 0.10)), T=T)
        states[("b16", T)] = reach_layer6(criteria, rbc, start_node,
                                           budget=16, T=T)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    plot_panel(axes[0, 0], states[("pct_10", 1.0)],
               "ERA pct_10 (budget=102), softmax T=1", "#1f77b4")
    plot_panel(axes[0, 1], states[("pct_10", 0.5)],
               "ERA pct_10 (budget=102), softmax T=0.5", "#7fbfff")
    plot_panel(axes[1, 0], states[("b16", 1.0)],
               "ERA b=16, softmax T=1", "#9467bd")
    plot_panel(axes[1, 1], states[("b16", 0.5)],
               "ERA b=16, softmax T=0.5", "#cea3f0")

    fig.suptitle("qid 51843, k=10, layer 6 (where BFS's CF lives) — "
                 "CF nodes shown in GREEN", fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = f"{OUT_DIR}/qid_51843_layer6_T_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    print("\nLayer 6 summary by config:")
    print(f"  {'config':<25} | {'kept':>6} {'pruned':>7} | {'CFs kept':>9} {'CFs pruned':>11}")
    for key, st in states.items():
        cfg = f"{key[0]}, T={key[1]}"
        nodes = st["unknown_nodes"]
        sorted_nodes = sorted(nodes, key=lambda n: n[3], reverse=True)
        cf_keys = st["cf_keys"]
        is_cf = [n[1] in cf_keys for n in sorted_nodes]
        ents = [n[3] for n in sorted_nodes]
        gaps = [ents[i] - ents[i+1] for i in range(len(ents) - 1)]
        max_gap_idx = max(range(len(gaps)), key=lambda i: (gaps[i], i))
        cutoff = max_gap_idx + 1
        kept = cutoff
        pruned = len(ents) - cutoff
        cf_kept = sum(is_cf[:cutoff])
        cf_pruned = sum(is_cf[cutoff:])
        print(f"  {cfg:<25} | {kept:>6} {pruned:>7} | {cf_kept:>9} {cf_pruned:>11}")


if __name__ == "__main__":
    main()
