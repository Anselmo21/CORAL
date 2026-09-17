"""Emit the paper's Table I and Table II from the GERA suite and config sweep.

Table I: every method against BFS at the operating budget, one row per model
and dataset. Bold marks the best accuracy per row and the lowest cost among
the pruning arms ERA/GRA/GERA (SRA and LSM sit outside the cost bolding: SRA
is near-free by construction and would sweep it). Table II: GERA under each
entropy scorer and rank aggregation method, macro-averaged over the models,
best accuracy per dataset column in bold. The pruning block reports the arms
on Qwen3-32B under the acronyms they reduce to (GRA drops the entropy cut,
ERA drops the graph).

Reads results/gera_suite.json and results/gera_config.json. Writes
results/paper_figs/gera/{merged_table,config_table,ablation_block}.tex.

Usage:
    python3 -m scripts.shared.build_gera_tables
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "results" / "gera_suite.json"
CONFIG = REPO / "results" / "gera_config.json"
OUT_DIR = REPO / "results" / "paper_figs" / "gera"

BUDGET = "16"
TABLE_MODELS = [
    ("llama31-8b-regen", "Llama3.1-8B"),
    ("llama31-70b-regen", "Llama3.1-70B"),
    ("qwen35-122b-regen", "Qwen3.5-122B"),
]
DS_ORDER = [
    ("msmarco_criteria", "msmarco criteria"),
    ("curated_criteria", "curated criteria"),
    ("msmarco_pragma", "msmarco pragmas"),
    ("curated_pragma", "curated pragmas"),
]
CONFIG_BLOCKS = [
    ("entropy", "Entropy", [
        ("softmax", "Softmax (default)"),
        ("shannon", "Shannon"),
        ("weighted", "Weighted"),
        ("renyi_0.5", "R\\'enyi ($\\alpha{=}0.5$)"),
        ("renyi_2", "R\\'enyi ($\\alpha{=}2$)"),
    ]),
    ("aggregator", "Rank\\\\Aggregator", [
        ("borda", "Borda (default)"),
        ("kemeny", "Kemeny"),
        ("mct", "MCT"),
        ("mc4", "MC4"),
        ("outranking", "Outranking"),
    ]),
]
ABLATION_MODEL = "qwen3-32b-regen"
ABLATION_ARMS = [
    ("gera", "GERA (default)"),
    ("no_entropy", "GRA"),
    ("no_graph", "ERA"),
    ("got_only", "GoT only"),
]


def fmt(x, bold=False):
    s = f"{x:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def cell(stats, ds, arm):
    return stats[ds]["ablations"][arm] if arm in stats[ds]["ablations"] else None


def build_merged_table(suite, budget=BUDGET):
    """Every method against BFS in one full-width table: the baselines and the
    pruning arms side by side, one budget, model group rows, GoT budget-less."""
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{@{}ll rrrrrr rrrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{6}{c}{Accuracy (\\%)} & \\multicolumn{6}{c}{Average Cost (LLM Calls)} \\\\",
        "\\cmidrule(lr){3-8} \\cmidrule(lr){9-14}",
        "Model & Dataset & SRA & LSM & ERA & GRA & GERA & GoT"
        " & BFS & SRA & LSM & ERA & GRA & GERA \\\\",
        "\\midrule\\addlinespace[1.5pt]",
    ]
    for mi, (model, nice) in enumerate(ALL_MODELS):
        stats = suite[model]
        if mi:
            lines.append("\\midrule\\addlinespace[1.5pt]")
        lines.append(f"\\multirow{{4}}{{*}}{{\\texttt{{{nice}}}}}")
        for ds, ds_nice in DS_ORDER:
            s = stats[ds]
            sweep = s["sweep"][budget]
            lsm = s["lsm"][budget]
            got = s["ablations"]["got_only"]
            cols = [s["sra"], lsm, sweep["era"], sweep["gra"], sweep["gera"]]
            best = max(c["accuracy_pct"] for c in cols)
            accs = [fmt(c["accuracy_pct"], bold=(c["accuracy_pct"] == best))
                    for c in cols] + [f"{got['accuracy_pct']:.1f}"]
            arm_best = min(c["mean_calls"] for c in cols[2:])
            costs = ([f"{s['bfs_calls']:.1f}"]
                     + [f"{c['mean_calls']:.1f}" for c in cols[:2]]
                     + [fmt(c["mean_calls"], bold=(c["mean_calls"] == arm_best))
                        for c in cols[2:]])
            lines.append(f" & {ds_nice} & " + " & ".join(accs)
                         + " & " + " & ".join(costs) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Accuracy and average cost of every search strategy across the"
        f" four LLMs, with $b{{=}}{budget}$ for the budgeted methods. Bold marks the"
        " best accuracy and the lowest cost among ERA, GRA, and GERA.}",
        "\\label{tab:cost-savings}\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_config_table(config):
    """GERA under each entropy scorer and rank aggregation method, one row per
    variant, macro-averaged over the models so no single LLM picks the winner."""
    cells = config["cells"]
    models = [m for m, _ in ALL_MODELS]

    def mean(block, variant, ds, field):
        return sum(cells[m][ds][block][variant][field] for m in models) / len(models)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{ll rr rr rr rr}",
        "\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nice}}}" for _, nice in DS_ORDER) + " \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}",
        "Setting & Variant" + " & Accuracy & Cost" * len(DS_ORDER) + " \\\\",
    ]
    for bi, (block, nice_block, variants) in enumerate(CONFIG_BLOCKS):
        lines.append("\\midrule" if not bi else "\\midrule\\addlinespace[1.5pt]")
        lines.append(f"\\multirow{{{len(variants)}}}{{*}}{{\\shortstack[l]{{{nice_block}}}}}")
        best = {ds: max(round(mean(block, v, ds, "accuracy_pct"), 1) for v, _ in variants)
                for ds, _ in DS_ORDER}
        for variant, nice in variants:
            row = []
            for ds, _ in DS_ORDER:
                acc = round(mean(block, variant, ds, "accuracy_pct"), 1)
                row.append(fmt(acc, bold=(acc == best[ds])))
                row.append(f"{mean(block, variant, ds, 'mean_calls'):.1f}")
            lines.append(f" & {nice} & " + " & ".join(row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{GERA ($b{{=}}{config['budget']}$) under each entropy scorer and rank"
        " aggregation method, averaged over the four LLMs; cost is the average"
        " number of LLM calls.}",
        "\\label{tab:ablations}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_ablation_block(suite):
    stats = suite[ABLATION_MODEL]
    cols = ["msmarco_criteria", "curated_criteria", "msmarco_pragma", "curated_pragma"]
    best = {ds: max(stats[ds]["ablations"][a]["accuracy_pct"] for a, _ in ABLATION_ARMS)
            for ds in cols}
    lines = ["\\midrule\\addlinespace[1.5pt]",
             "\\multirow{4}{*}{\\shortstack[l]{Pruning\\\\(GERA $b{=}4$, \\texttt{Qwen3-32B})}}"]
    for arm, nice in ABLATION_ARMS:
        cells = []
        for ds in cols:
            c = stats[ds]["ablations"][arm]
            cells.append(fmt(c["accuracy_pct"], bold=(c["accuracy_pct"] == best[ds])))
            cells.append(f"{c['mean_calls']:.1f}")
        lines.append(f" & {nice} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


ALL_MODELS = [
    ("llama31-8b-regen", "Llama3.1-8B"),
    ("llama31-70b-regen", "Llama3.1-70B"),
    ("qwen3-32b-regen", "Qwen3-32B"),
    ("qwen35-122b-regen", "Qwen3.5-122B"),
]
PRUNING_BUDGETS = ("4", "8", "16")


def build_pruning_table(suite):
    """The pruning ablation at the operating budget, Table I's granularity and
    layout: one row per dataset under a model header, GoT alone budget-less."""
    b = "16"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{0.96}",
        "\\begin{tabular}{@{}l rrrr rrrr@{}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Accuracy (\\%)} & \\multicolumn{4}{c}{Average Cost (LLM Calls)} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "Dataset & ERA & GRA & GERA & GoT & ERA & GRA & GERA & GoT \\\\",
    ]
    for model, nice in ALL_MODELS:
        stats = suite[model]
        lines.append("\\midrule\\addlinespace[1.5pt]")
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\texttt{{{nice}}}}} \\\\")
        for ds, ds_nice in DS_ORDER:
            s = stats[ds]
            sweep = s["sweep"][b]
            got = s["ablations"]["got_only"]
            best = max(sweep[m]["accuracy_pct"] for m in ("era", "gra", "gera"))
            accs = [fmt(sweep[m]["accuracy_pct"], bold=(sweep[m]["accuracy_pct"] == best))
                    for m in ("era", "gra", "gera")] + [f"{got['accuracy_pct']:.1f}"]
            costs = [f"{sweep[m]['mean_calls']:.1f}" for m in ("era", "gra", "gera")] \
                    + [f"{got['mean_calls']:.1f}"]
            lines.append(f"\\;\\;{ds_nice} & " + " & ".join(accs)
                         + " & " + " & ".join(costs) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The pruning arms at $b{=}16$: ERA (no GoT), GRA (no entropy"
        " cut), GERA, and the GoT proposals alone.}",
        "\\label{tab:pruning-ablation}\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "merged_table.tex").write_text(build_merged_table(suite), encoding="utf-8")
    (OUT_DIR / "ablation_block.tex").write_text(build_ablation_block(suite), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'merged_table.tex'}")
    print(f"wrote {OUT_DIR / 'ablation_block.tex'}")
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        (OUT_DIR / "config_table.tex").write_text(build_config_table(config), encoding="utf-8")
        print(f"wrote {OUT_DIR / 'config_table.tex'}")


if __name__ == "__main__":
    main()
