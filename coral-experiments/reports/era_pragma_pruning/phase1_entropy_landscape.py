"""Visualize the entropy landscape that drives ERA Phase 1 allocation.

For each Phase 1 layer (L2..L_{k-1}), show the sorted entropy values and the
max_jump_cutoff position. Compare a pragma case (sonnet_pragma qid=6 k=10,
where pruning fails) against a criteria case (sonnet_subsample qid=2 k=10,
where pruning works well).

Output: reports/era_pragma_pruning/figures/phase1_entropy_landscape.png

Backs the claim that on pragma, per-layer entropy is "flat" — span is tiny,
ranking is dominated by floating-point noise, so max_jump_cutoff picks
nearly-arbitrary nodes for lookup. On criteria, entropy has a real cliff that
max_jump can find.
"""
from __future__ import annotations

import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.pragma.sonnet.run_subsample import _load_pragma_lattices
from scripts.shared.plot_style import apply_plot_style
from scripts.sonnet.run_subsample import _load_lattices as _load_sonnet_criteria
from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key
from search.rank import _aggregate_with_scores, _get_child_rankings

OUT_DIR = "reports/era_pragma_pruning/figures"
BUDGET  = 16


def phase1_entropies(criteria: list[str], rbc: dict[str, list[str]],
                     start_node: list[str], budget: int) -> list[dict]:
    """Replay ERA Phase 1 layer-by-layer and capture per-layer entropies."""
    answers = list(rbc[combo_key(start_node)])
    s = len(start_node)
    known = {c: rbc[c] for c in criteria}
    estimated: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}
    budget_rem = budget

    per_layer: list[dict] = []
    for layer_size in range(2, s):
        layer_nodes: list[tuple[str, float]] = []
        for combo in combinations(range(s), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)
            voters = _get_child_rankings(subset, known, estimated)
            if not voters:
                continue
            ranking, scores = _aggregate_with_scores(voters, answers, "combsum")
            if not scores:
                continue
            estimated[key] = ranking
            entropies[key] = softmax_entropy(scores, num_voters=len(voters))
            layer_nodes.append((key, entropies[key]))

        ents = [e for _, e in layer_nodes]
        # max_jump cutoff position
        if len(ents) > 1:
            sorted_ents = sorted(ents, reverse=True)
            gaps = [sorted_ents[i] - sorted_ents[i+1] for i in range(len(sorted_ents) - 1)]
            cutoff_idx = max(range(len(gaps)), key=lambda j: (gaps[j], j))
            max_gap   = gaps[cutoff_idx]
            cutoff_pos = cutoff_idx + 1
        else:
            sorted_ents, max_gap, cutoff_pos = ents, 0.0, 1

        selected = cutoff_fn_or_cap(layer_nodes, budget_rem)
        for key in selected:
            known[key] = rbc[key]
            budget_rem -= 1

        per_layer.append({
            "layer": layer_size,
            "ents_sorted_desc": sorted(ents, reverse=True),
            "span":           max(ents) - min(ents) if ents else 0.0,
            "max_gap":        max_gap,
            "cutoff_pos":     cutoff_pos,
            "n":              len(ents),
            "selected_for_lookup": len(selected),
        })
    return per_layer


def cutoff_fn_or_cap(layer_nodes: list[tuple[str, float]], budget_rem: int) -> list[str]:
    if budget_rem <= 0:
        return []
    return max_jump_cutoff(layer_nodes, budget_rem)


def plot_landscape(pragma: list[dict], criteria: list[dict],
                   pragma_label: str, criteria_label: str, out: str) -> None:
    """Row 1 = pragma per-layer entropy curves; Row 2 = criteria same."""
    n_layers = max(len(pragma), len(criteria))
    fig, axes = plt.subplots(2, n_layers, figsize=(2.2 * n_layers, 5.4),
                             squeeze=False, sharey="row")

    for row_idx, (data, label, color) in enumerate(
        [(pragma, pragma_label, "#9467bd"),
         (criteria, criteria_label, "#1f77b4")]
    ):
        for j, info in enumerate(data):
            ax = axes[row_idx, j]
            ents = info["ents_sorted_desc"]
            x = np.arange(len(ents))
            ax.bar(x, ents, color=color, alpha=0.75, width=1.0, edgecolor="none")
            ax.axvline(info["cutoff_pos"] - 0.5, color="red", linestyle="--", linewidth=1.0)
            ax.set_title(
                f"L{info['layer']}  n={info['n']}\n"
                f"span={info['span']:.4f}  max-gap={info['max_gap']:.4f}",
                fontsize=8,
            )
            ax.set_xlabel("nodes ↓ ent")
            if j == 0:
                ax.set_ylabel(f"{label}\nsoftmax H")
            ax.grid(True, alpha=0.3, axis="y")
        for j in range(len(data), n_layers):
            axes[row_idx, j].axis("off")

    fig.suptitle(
        "Phase 1 entropy landscape — sorted desc with max_jump cutoff (red dashed)\n"
        f"top: {pragma_label} (flat — span ≪ 1)        "
        f"bottom: {criteria_label} (cliff — visible elbow)",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # PRAGMA case: qid=6 sonnet pragma, k=10
    lats = _load_pragma_lattices(["6"])
    p_crit, p_rbc = lats["6"]
    p_layers = phase1_entropies(p_crit, p_rbc, list(p_crit), BUDGET)

    # CRITERIA case: sonnet_subsample qid=2 (or first available), k=10
    # use the same loader as run_subsample
    qids = ["20"]
    c_lats = _load_sonnet_criteria(qids)
    qid_c = list(c_lats.keys())[0]
    c_crit, c_rbc = c_lats[qid_c]
    c_layers = phase1_entropies(c_crit, c_rbc, list(c_crit), BUDGET)

    print("PRAGMA  sonnet qid=6 k=10:")
    for info in p_layers:
        print(f"  L{info['layer']}  n={info['n']:>3}  span={info['span']:.5f}  "
              f"max_gap={info['max_gap']:.5f}  cutoff={info['cutoff_pos']:>2}  "
              f"selected={info['selected_for_lookup']}")
    print()
    print(f"CRITERIA  sonnet qid={qid_c} k=10:")
    for info in c_layers:
        print(f"  L{info['layer']}  n={info['n']:>3}  span={info['span']:.5f}  "
              f"max_gap={info['max_gap']:.5f}  cutoff={info['cutoff_pos']:>2}  "
              f"selected={info['selected_for_lookup']}")

    out = f"{OUT_DIR}/phase1_entropy_landscape.png"
    plot_landscape(p_layers, c_layers,
                   "Pragma (Sonnet qid=6)", f"Criteria (Sonnet qid={qid_c})", out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
