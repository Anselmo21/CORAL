"""Plot per-variant evaluation results produced by evaluate_graphs.py.

Reads graph_eval_{criteria,pragma}_cf_removal.json and writes PNGs to plots/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
PLOTS = REPO / "plots"
PLOTS.mkdir(exist_ok=True)

VARIANTS = {
    "criteria": REPO / "graph_eval_criteria_cf_removal.json",
    "pragma": REPO / "graph_eval_pragma_cf_removal.json",
}

COLORS = {"criteria": "#3b7dd8", "pragma": "#d8733b"}


def load(path: Path):
    data = json.loads(path.read_text())
    return [r for r in data
            if r["truth"]["lattice_complete"]
            and r["truth"]["full_top1_matches_baseline"] is True
            and r["got_min"]]


datasets = {k: load(v) for k, v in VARIANTS.items()}


# ---------- Plot 1: |S_got| vs |S_true| scatter, one panel per variant ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, (name, rs) in zip(axes, datasets.items()):
    xs = [r["got_min"]["retain_size"] for r in rs]
    ys = [r["truth"]["max_retain_size"] for r in rs]
    n_atoms = [r["num_atoms"] for r in rs]
    ax.scatter(xs, ys, alpha=0.55, s=60, c=COLORS[name], edgecolors="black", linewidth=0.5)
    lo = 0
    hi = max(max(xs, default=0), max(ys, default=0), max(n_atoms, default=0)) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="|S_got| = |S_true|")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("|S_got| (GoT largest-S)")
    ax.set_ylabel("|S_true| (BFS max-flip)")
    ax.set_title(f"{name}  (n={len(rs)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("|S_got| vs |S_true| per query")
fig.tight_layout()
fig.savefig(PLOTS / "01_size_scatter.png", dpi=140)
plt.close(fig)


# ---------- Plot 2: Size-gap histogram (|S_got| - |S_true|) ----------
fig, ax = plt.subplots(figsize=(9, 5))
all_gaps = []
for name, rs in datasets.items():
    gaps = [r["got_min"]["vs_true_size_gap"] for r in rs]
    all_gaps.append((name, gaps))
mn = min(min(g) for _, g in all_gaps)
mx = max(max(g) for _, g in all_gaps)
bins = np.arange(mn - 0.5, mx + 1.5, 1)
for name, gaps in all_gaps:
    ax.hist(gaps, bins=bins, alpha=0.55, label=f"{name} (n={len(gaps)})",
            color=COLORS[name], edgecolor="black", linewidth=0.5)
ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("|S_got| − |S_true|   (>0 = GoT retains more; <0 = GoT over-removes)")
ax.set_ylabel("# queries")
ax.set_title("Size gap between GoT's largest-S and true max-flip S")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "02_size_gap_hist.png", dpi=140)
plt.close(fig)


# ---------- Plot 3: Q1 categorical bar (match / under-remove / over-remove) ----------
fig, ax = plt.subplots(figsize=(9, 5))
cats = ["|S_got| = |S_true|\n(size match)",
        "|S_true| < |S_got|\n(GoT under-removes)",
        "|S_true| > |S_got|\n(GoT over-removes)"]
x = np.arange(len(cats))
width = 0.38
for i, (name, rs) in enumerate(datasets.items()):
    n = len(rs)
    gaps = [r["got_min"]["vs_true_size_gap"] for r in rs]
    counts = [sum(g == 0 for g in gaps), sum(g > 0 for g in gaps), sum(g < 0 for g in gaps)]
    pcts = [c / n * 100 for c in counts]
    bars = ax.bar(x + (i - 0.5) * width, pcts, width, label=f"{name} (n={n})",
                  color=COLORS[name], edgecolor="black", linewidth=0.5)
    for b, c, p in zip(bars, counts, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{c}\n{p:.0f}%",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("% of queries")
ax.set_ylim(0, 85)
ax.set_title("Q1: does |S_got| match |S_true|?")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "03_q1_size_agreement.png", dpi=140)
plt.close(fig)


# ---------- Plot 4: Q2 histogram of levels below (only cases where true CF is lower) ----------
fig, ax = plt.subplots(figsize=(9, 5))
series = []
for name, rs in datasets.items():
    lb = [r["got_min"]["vs_true_size_gap"] for r in rs if r["got_min"]["vs_true_size_gap"] > 0]
    series.append((name, lb))
mx = max((max(s) for _, s in series if s), default=1)
bins = np.arange(0.5, mx + 1.5, 1)
for name, lb in series:
    ax.hist(lb, bins=bins, alpha=0.55, label=f"{name} (n={len(lb)})",
            color=COLORS[name], edgecolor="black", linewidth=0.5)
ax.set_xlabel("|S_got| − |S_true|  (lattice levels below S_got)")
ax.set_ylabel("# queries")
ax.set_title("Q2: how many levels below S_got is S_true?\n(only queries where true CF is smaller)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "04_q2_levels_below.png", dpi=140)
plt.close(fig)


# ---------- Plot 5: Q3 containment bars ----------
fig, ax = plt.subplots(figsize=(9, 5))
cats = ["S_got ⊇ some MAX-flip S_true", "S_got contains ANY flip subset"]
x = np.arange(len(cats))
width = 0.38
for i, (name, rs) in enumerate(datasets.items()):
    smaller = [r for r in rs if r["got_min"]["vs_true_size_gap"] > 0]
    n = len(smaller)
    if n == 0:
        continue
    a = sum(1 for r in smaller if r["got_min"]["vs_true_contains_true_max_cf"])
    b = sum(1 for r in smaller if r["got_min"]["vs_true_largest_flipping_subset_size"] is not None)
    pcts = [a / n * 100, b / n * 100]
    bars = ax.bar(x + (i - 0.5) * width, pcts, width, label=f"{name} (n={n})",
                  color=COLORS[name], edgecolor="black", linewidth=0.5)
    for b_, cnt, p in zip(bars, [a, b], pcts):
        ax.text(b_.get_x() + b_.get_width() / 2, p + 1, f"{cnt}\n{p:.0f}%",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("% of queries with |S_true| < |S_got|")
ax.set_ylim(0, 115)
ax.set_title("Q3: is a true CF reachable by removing concepts from S_got?")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "05_q3_containment.png", dpi=140)
plt.close(fig)


# ---------- Plot 6: Jaccard distribution (got_min vs closest true S) ----------
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 1, 11)
for name, rs in datasets.items():
    js = [r["got_min"]["vs_true_jaccard"] for r in rs if r["got_min"]["vs_true_jaccard"] is not None]
    ax.hist(js, bins=bins, alpha=0.55, label=f"{name} (n={len(js)})",
            color=COLORS[name], edgecolor="black", linewidth=0.5)
ax.set_xlabel("Jaccard distance to closest true S_true  (0 = identical, 1 = disjoint)")
ax.set_ylabel("# queries")
ax.set_title("GoT largest-S vs closest true S_true — Jaccard distance")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "06_jaccard_hist.png", dpi=140)
plt.close(fig)


# ---------- Plot 7: validity / target-match / any-flip bars ----------
fig, ax = plt.subplots(figsize=(9, 5))
cats = ["GoT best-S_i\nflips top-1", "GoT best-S_i\nhits target", "≥1 S_i\nflips top-1"]
x = np.arange(len(cats))
width = 0.38
for i, (name, rs) in enumerate(datasets.items()):
    n = len(rs)
    valid = sum(1 for r in rs if r["got_min"]["flip_success"])
    tgt = sum(1 for r in rs if r["got_min"]["target_match"])
    any_v = sum(1 for r in rs if r["got_best_valid"])
    counts = [valid, tgt, any_v]
    pcts = [c / n * 100 for c in counts]
    bars = ax.bar(x + (i - 0.5) * width, pcts, width, label=f"{name} (n={n})",
                  color=COLORS[name], edgecolor="black", linewidth=0.5)
    for b, c, p in zip(bars, counts, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{c}\n{p:.0f}%",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("% of queries")
ax.set_ylim(0, 80)
ax.set_title("GoT counterfactual validity in the lattice")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "07_validity.png", dpi=140)
plt.close(fig)


# ---------- Plot 8: diverging over/match/under bar ----------
fig, ax = plt.subplots(figsize=(10, 3.6))
y_positions = np.arange(len(datasets))
labels = list(datasets.keys())

over_pcts, match_pcts, under_pcts = [], [], []
over_counts, match_counts, under_counts = [], [], []
ns = []
for name, rs in datasets.items():
    n = len(rs)
    ns.append(n)
    gaps = [r["got_min"]["vs_true_size_gap"] for r in rs]
    over = sum(g < 0 for g in gaps)     # |S_true| > |S_got|  → GoT removed too many
    match = sum(g == 0 for g in gaps)
    under = sum(g > 0 for g in gaps)    # |S_true| < |S_got|  → GoT removed too few
    over_counts.append(over); match_counts.append(match); under_counts.append(under)
    over_pcts.append(over / n * 100)
    match_pcts.append(match / n * 100)
    under_pcts.append(under / n * 100)

C_OVER = "#c94a4a"   # over-remove (removed too many)
C_MATCH = "#9c9c9c"  # exact
C_UNDER = "#4a8fc9"  # under-remove (removed too few)

# Draw over-remove to the left of 0, match centered on 0, under-remove to the right.
# Match spans [-match/2, +match/2] so 0 is the midpoint of the neutral zone.
for i, name in enumerate(labels):
    over = over_pcts[i]
    match = match_pcts[i]
    under = under_pcts[i]
    ax.barh(i, -over, left=-match/2, color=C_OVER, edgecolor="black", linewidth=0.5,
            label="Over-remove (|S_true| > |S_got|)" if i == 0 else None)
    ax.barh(i, match, left=-match/2, color=C_MATCH, edgecolor="black", linewidth=0.5,
            label="Exact match" if i == 0 else None)
    ax.barh(i, under, left=match/2, color=C_UNDER, edgecolor="black", linewidth=0.5,
            label="Under-remove (|S_true| < |S_got|)" if i == 0 else None)
    # Labels
    ax.text(-match/2 - over/2, i, f"{over_counts[i]}\n{over:.0f}%",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(0, i, f"{match_counts[i]}\n{match:.0f}%",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(match/2 + under/2, i, f"{under_counts[i]}\n{under:.0f}%",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")

ax.set_yticks(y_positions)
ax.set_yticklabels([f"{n}\n(n={ns[i]})" for i, n in enumerate(labels)])
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("% of queries  ← over-remove  |  exact  |  under-remove →")
ax.set_title("Overshoot vs undershoot: where does GoT's minimal CF land relative to S_true?")
ax.set_xlim(-100, 100)
ax.set_xticks(np.arange(-100, 101, 25))
ax.set_xticklabels([f"{abs(x)}%" for x in np.arange(-100, 101, 25)])
ax.grid(True, axis="x", alpha=0.3)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / "08_overshoot_undershoot.png", dpi=140, bbox_inches="tight")
plt.close(fig)


print(f"Wrote 8 plots to {PLOTS}")
for p in sorted(PLOTS.glob("*.png")):
    print(" -", p.name)
