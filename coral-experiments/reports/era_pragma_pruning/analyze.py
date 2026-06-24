"""Aggregate the per-layer instrumentation, compute summary stats, generate figures.

Inputs:
  reports/era_pragma_pruning/data/layers.jsonl   (produced by instrument.py)

Outputs:
  reports/era_pragma_pruning/data/summary.json
  reports/era_pragma_pruning/figures/
    1_entropy_distribution.png       — per-dataset entropy CDF of every node entropy seen
    2_layer_entropy_range.png        — per-dataset distribution of within-layer entropy range
    3_cutoff_fraction.png            — fraction of layer classified high-entropy (cutoff selects)
    4_max_gap_fraction.png           — max gap as fraction of layer entropy range
    5_pruning_rate.png               — pruned / displacement_0 per layer per dataset
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

LAYERS_PATH = "reports/era_pragma_pruning/data/layers.jsonl"
SUMMARY_PATH = "reports/era_pragma_pruning/data/summary.json"
FIG_DIR = "reports/era_pragma_pruning/figures"

DATASETS = ["msmarco_criteria", "sonnet_criteria", "msmarco_pragma", "sonnet_pragma"]
DISPLAY = {
    "msmarco_criteria": "MSMARCO (criteria)",
    "sonnet_criteria":  "SONNET (criteria)",
    "msmarco_pragma":   "MSMARCO (pragma)",
    "sonnet_pragma":    "SONNET (pragma)",
}
COLORS = {
    "msmarco_criteria": "#1f77b4",
    "sonnet_criteria":  "#2ca02c",
    "msmarco_pragma":   "#d62728",
    "sonnet_pragma":    "#ff7f0e",
}


def load_layers() -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    with open(LAYERS_PATH) as f:
        for line in f:
            r = json.loads(line)
            by_ds[r["dataset"]].append(r)
    return dict(by_ds)


def cdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.array(values, dtype=float))
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y


def plot_entropy_distribution(by_ds: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for ds in DATASETS:
        ent = [e for r in by_ds[ds] for e in r["entropy_values"] if np.isfinite(e)]
        x, y = cdf(ent)
        ax.plot(x, y, label=DISPLAY[ds], color=COLORS[ds], linewidth=2)
    ax.set_xlabel("Per-node softmax entropy (bits)")
    ax.set_ylabel("CDF over all displacement / unknown nodes")
    ax.set_title("Phase-2 node entropy distribution (k=8, 20 cases / dataset)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/1_entropy_distribution.png", dpi=150)
    plt.close(fig)


def plot_layer_entropy_range(by_ds: dict[str, list[dict]]) -> None:
    """Within each layer evaluation: max - min entropy. Tells us how compressed the
    entropy distribution is. Compressed = max_jump_cutoff has nothing to grip onto."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for ds in DATASETS:
        ranges = []
        for r in by_ds[ds]:
            e = [v for v in r["entropy_values"] if np.isfinite(v)]
            if len(e) >= 2:
                ranges.append(max(e) - min(e))
        data.append(ranges)
        labels.append(DISPLAY[ds])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, ds in zip(bp["boxes"], DATASETS):
        patch.set_facecolor(COLORS[ds])
        patch.set_alpha(0.6)
    ax.set_ylabel("Within-layer entropy range  (max − min, bits)")
    ax.set_title("How spread-out is the entropy distribution inside one layer?")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/2_layer_entropy_range.png", dpi=150)
    plt.close(fig)


def plot_cutoff_fraction(by_ds: dict[str, list[dict]]) -> None:
    """n_high_entropy / n_unknown — fraction of the layer that the cutoff
    classifies as high-entropy (= cannot be pruned). 1.0 means the cutoff
    selected everyone → pruning impossible."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for ds in DATASETS:
        frac = [r["n_high_entropy"] / r["n_unknown"] for r in by_ds[ds] if r["n_unknown"] > 0]
        data.append(frac)
        labels.append(DISPLAY[ds])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, ds in zip(bp["boxes"], DATASETS):
        patch.set_facecolor(COLORS[ds])
        patch.set_alpha(0.6)
    ax.axhline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.7,
               label="cutoff keeps every node (no pruning possible)")
    ax.set_ylabel("|high-entropy| / |unknown nodes|")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Cutoff classification: where does max_jump_cutoff place the boundary?")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/3_cutoff_fraction.png", dpi=150)
    plt.close(fig)


def plot_max_gap_fraction(by_ds: dict[str, list[dict]]) -> None:
    """max_gap / entropy_range — how 'meaningful' is the largest gap relative to
    the spread? If all gaps are tiny and roughly equal, the tie-break (last index)
    fires and the cutoff selects every node."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for ds in DATASETS:
        fracs = [r["max_gap_frac"] for r in by_ds[ds]
                 if r["max_gap_frac"] is not None and not (isinstance(r["max_gap_frac"], float) and np.isnan(r["max_gap_frac"]))]
        data.append(fracs)
        labels.append(DISPLAY[ds])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, ds in zip(bp["boxes"], DATASETS):
        patch.set_facecolor(COLORS[ds])
        patch.set_alpha(0.6)
    ax.set_ylabel("max_gap / (max_entropy − min_entropy)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("How distinct is the largest entropy gap inside one layer?")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/4_max_gap_fraction.png", dpi=150)
    plt.close(fig)


def plot_pruning_rate(by_ds: dict[str, list[dict]]) -> None:
    """n_pruned / n_disp_0 — among the nodes that ARE prunable in principle
    (estimate kept baseline at top-1), what fraction did the cutoff actually
    prune?"""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []
    for ds in DATASETS:
        rates = []
        for r in by_ds[ds]:
            if r["n_disp_0"] > 0:
                rates.append(r["n_pruned"] / r["n_disp_0"])
        data.append(rates)
        labels.append(DISPLAY[ds])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, ds in zip(bp["boxes"], DATASETS):
        patch.set_facecolor(COLORS[ds])
        patch.set_alpha(0.6)
    ax.set_ylabel("n_pruned / n_displacement_0")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Realized pruning rate among displacement-0 (prunable) nodes")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/5_pruning_rate.png", dpi=150)
    plt.close(fig)


def compute_summary(by_ds: dict[str, list[dict]]) -> dict:
    summary: dict[str, dict] = {}
    for ds in DATASETS:
        rows = by_ds[ds]
        all_ent = [e for r in rows for e in r["entropy_values"] if np.isfinite(e)]
        finite_layers = [[v for v in r["entropy_values"] if np.isfinite(v)] for r in rows]
        layer_ranges = [max(e) - min(e) for e in finite_layers if len(e) >= 2]
        cutoff_fracs = [r["n_high_entropy"] / r["n_unknown"] for r in rows if r["n_unknown"] > 0]
        gap_fracs = [r["max_gap_frac"] for r in rows
                     if r["max_gap_frac"] is not None
                     and not (isinstance(r["max_gap_frac"], float) and np.isnan(r["max_gap_frac"]))]
        pruning_rates = [r["n_pruned"] / r["n_disp_0"] for r in rows if r["n_disp_0"] > 0]

        summary[ds] = {
            "n_layers": len(rows),
            "n_total_unknowns": sum(r["n_unknown"] for r in rows),
            "n_total_disp_0": sum(r["n_disp_0"] for r in rows),
            "n_total_pruned": sum(r["n_pruned"] for r in rows),
            "entropy_mean": float(np.mean(all_ent)),
            "entropy_median": float(np.median(all_ent)),
            "entropy_std": float(np.std(all_ent)),
            "layer_range_mean": float(np.mean(layer_ranges)) if layer_ranges else None,
            "layer_range_median": float(np.median(layer_ranges)) if layer_ranges else None,
            "cutoff_frac_median": float(np.median(cutoff_fracs)) if cutoff_fracs else None,
            "cutoff_frac_mean": float(np.mean(cutoff_fracs)) if cutoff_fracs else None,
            "cutoff_frac_eq_1_rate": float(np.mean([f == 1.0 for f in cutoff_fracs])) if cutoff_fracs else None,
            "max_gap_frac_median": float(np.median(gap_fracs)) if gap_fracs else None,
            "pruning_rate_median": float(np.median(pruning_rates)) if pruning_rates else None,
            "pruning_rate_mean": float(np.mean(pruning_rates)) if pruning_rates else None,
        }
    return summary


def main() -> None:
    apply_plot_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    by_ds = load_layers()

    plot_entropy_distribution(by_ds)
    plot_layer_entropy_range(by_ds)
    plot_cutoff_fraction(by_ds)
    plot_max_gap_fraction(by_ds)
    plot_pruning_rate(by_ds)

    summary = compute_summary(by_ds)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    for ds in DATASETS:
        s = summary[ds]
        print(f"\n  {DISPLAY[ds]} (n_layers={s['n_layers']}, unknowns={s['n_total_unknowns']})")
        print(f"    entropy: median={s['entropy_median']:.3f} bits  std={s['entropy_std']:.3f}")
        print(f"    layer_range: median={s['layer_range_median']:.4f} bits")
        print(f"    cutoff_frac (high-entropy / unknowns): median={s['cutoff_frac_median']:.3f}  mean={s['cutoff_frac_mean']:.3f}")
        print(f"    layers where cutoff kept every node:    {s['cutoff_frac_eq_1_rate']*100:.1f}%")
        print(f"    max_gap_frac (gap / range): median={s['max_gap_frac_median']:.4f}")
        print(f"    pruning rate (pruned / disp_0): median={s['pruning_rate_median']:.3f}  mean={s['pruning_rate_mean']:.3f}")
        print(f"    totals: disp_0={s['n_total_disp_0']} pruned={s['n_total_pruned']}")

    print(f"\nSummary → {SUMMARY_PATH}")
    print(f"Figures → {FIG_DIR}/")


if __name__ == "__main__":
    main()
