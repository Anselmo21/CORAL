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
