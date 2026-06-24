"""Inspect Phase 2 at layer 6 for qid 51843 — show the entropy distribution
of all unknown nodes plus where max_jump_cutoff places its boundary.
"""
from __future__ import annotations

import os
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


def reach_layer6(criteria, rbc, start_node, budget):
    """Re-run ERA Phase 1, then compute layer-6 Phase 2 state. Returns
    (entropy_sorted_desc, max_gap_idx, baseline_top1_in_cf, cf_keys_set)."""
    aggregator = "combsum"
    entropy_fn = softmax_entropy
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

    # Now walk Phase 2 until we reach layer 6
    for layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        if layer_size > 6:
            # Update known with checked candidates
            unknown_nodes = []
            for subset in subsets:
                key = combo_key(subset)
                if key in known:
                    if known[key] and known[key][0] != baseline_answer:
                        return None   # would've returned wrong_layer earlier
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
                    return None   # CF found at deeper layer, would've returned wrong
        elif layer_size == 6:
            # Capture state at layer 6
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
                # Also tag if the GROUND-TRUTH (which we know) for this key is a CF
                gt = rbc.get(key)
                if gt and gt[0] != baseline_answer:
                    cf_keys.add(key)

            return {
                "unknown_nodes": unknown_nodes,
                "cf_keys": cf_keys,
                "n_known_here": sum(1 for s_ in subsets if combo_key(s_) in known),
                "total": len(subsets),
            }
    return None


def plot_layer6(state, label, color, ax_top, ax_bot):
    nodes = state["unknown_nodes"]
    cf_keys = state["cf_keys"]

    # Sort by entropy desc
    sorted_nodes = sorted(nodes, key=lambda n: n[3], reverse=True)
    entropies = [n[3] for n in sorted_nodes]
    keys      = [n[1] for n in sorted_nodes]
    is_cf     = [k in cf_keys for k in keys]

    # Compute cutoff
    gaps = [entropies[i] - entropies[i+1] for i in range(len(entropies) - 1)]
    max_gap_idx = max(range(len(gaps)), key=lambda i: (gaps[i], i)) if gaps else 0
    cutoff_pos = max_gap_idx + 1   # nodes 0..max_gap_idx are "high entropy"

    n_high = cutoff_pos
    n_pruned = len(entropies) - cutoff_pos

    # TOP PANEL: entropy distribution sorted
    x = np.arange(len(entropies))
    bar_colors = ["#2ca02c" if c else color for c in is_cf]   # CF nodes green
    ax_top.bar(x, entropies, color=bar_colors, width=1.0, alpha=0.75, edgecolor="none")
    ax_top.axvline(cutoff_pos - 0.5, color="red", linestyle="--", linewidth=1.5,
                   label=f"max_jump cutoff at idx {cutoff_pos} "
                         f"(high-ent: {n_high}, pruned: {n_pruned})")
    n_cf_above = sum(is_cf[:cutoff_pos])
    n_cf_below = sum(is_cf[cutoff_pos:])
    ax_top.set_title(
        f"{label} — layer 6 entropy distribution (sorted desc)  "
        f"|  {n_cf_above} CFs kept, {n_cf_below} CFs pruned",
        fontweight="bold", fontsize=11,
    )
    ax_top.set_ylabel("softmax entropy (bits)")
    ax_top.set_xlabel("nodes sorted by entropy (desc)")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(True, alpha=0.3, axis="y")

    # BOTTOM PANEL: gap sizes
    ax_bot.bar(np.arange(len(gaps)) + 0.5, gaps, color=color, width=1.0, alpha=0.75, edgecolor="none")
    ax_bot.axvline(max_gap_idx + 0.5, color="red", linestyle="--", linewidth=1.5,
                   label=f"max gap at idx {max_gap_idx}, size = {gaps[max_gap_idx]:.4f}")
    ax_bot.set_ylabel("gap to next node")
    ax_bot.set_xlabel("rank position")
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, alpha=0.3, axis="y")

    print(f"\n  {label}:")
    print(f"    Total layer-6 unknowns: {len(entropies)}")
    print(f"    Total layer-6 CFs (in unknowns): {sum(is_cf)}")
    print(f"    Entropy range: [{min(entropies):.5f}, {max(entropies):.5f}]   span: {max(entropies)-min(entropies):.5f}")
    print(f"    Max gap: {gaps[max_gap_idx]:.5f} at idx {max_gap_idx}  ({100*gaps[max_gap_idx]/(max(entropies)-min(entropies)+1e-12):.1f}% of range)")
    print(f"    Cutoff splits {len(entropies)} nodes into  high-entropy: {n_high}   low-entropy (pruned): {n_pruned}")
    print(f"    Of the {sum(is_cf)} real CFs in this layer:")
    print(f"      kept (above cutoff):  {n_cf_above}")
    print(f"      pruned (below cutoff): {n_cf_below}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    qid = "51843"
    lats = _load_lattices([qid])
    criteria, rbc = lats[qid]
    start_node = list(criteria)

    pct10 = reach_layer6(criteria, rbc, start_node, budget=max(1, int(2**10 * 0.10)))
    b16   = reach_layer6(criteria, rbc, start_node, budget=16)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    plot_layer6(pct10, "ERA pct_10 (budget=102)",  "#1f77b4", axes[0], axes[1])
    plot_layer6(b16,   "ERA b=16 (budget=16)",     "#9467bd", axes[2], axes[3])
    fig.suptitle("qid 51843, k=10 — Phase 2 at layer 6  (BFS finds CF here)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = f"{OUT_DIR}/qid_51843_layer6_entropy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
