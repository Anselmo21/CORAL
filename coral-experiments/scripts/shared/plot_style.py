"""Shared matplotlib style for report-ready figures.

Call `apply_plot_style()` at the start of every plot function so figures
remain readable when scaled down for the report.
"""


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "figure.titlesize": 16,
    })


def bootstrap_ci_mean(values, n_iter: int = 1000, ci: float = 0.95, seed: int = 0):
    """Mean + percentile bootstrap CI. Returns (mean, low, high) or (None,None,None)."""
    import numpy as np
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_iter, arr.size), replace=True).mean(axis=1)
    lo_p, hi_p = (1.0 - ci) / 2.0 * 100.0, (1.0 + ci) / 2.0 * 100.0
    low, high = np.percentile(boots, [lo_p, hi_p])
    return mean, float(low), float(high)


def draw_hop_bars(ax, sizes, method_data, handles_by_label, bar_width_total: float = 0.8):
    """Grouped bar chart for hop distance.

    method_data: list of (label, color, {size -> list[float]}). Methods with no
    values at any size are skipped silently.
    """
    n = len(method_data)
    width = bar_width_total / n
    for i, (lab, color, per_size) in enumerate(method_data):
        xs, means, lo_err, hi_err = [], [], [], []
        for s in sizes:
            m, lo, hi = bootstrap_ci_mean(per_size.get(s, []))
            if m is None:
                continue
            xs.append(s); means.append(m)
            lo_err.append(m - lo); hi_err.append(hi - m)
        if not xs:
            continue
        offsets = [x + (i - (n - 1) / 2.0) * width for x in xs]
        bars = ax.bar(offsets, means, width=width, color=color, label=lab,
                      yerr=[lo_err, hi_err], capsize=2,
                      edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.8})
        handles_by_label.setdefault(lab, bars[0])
