"""Emit the paper's Table I and the pruning-ablation block from gera_suite.json.

Table I: SRA, LSM and GERA at b=4 against BFS, one row per model and dataset,
Qwen3-32B left out (it has its own figure). Single-column layout: the model
names sit on their own header rows so the numbers fit the IEEE column. The
ablation block slots into tab:ablations under the entropy and rank-aggregator
blocks, reporting the pruning arms on Qwen3-32B; the arms carry the acronyms
they reduce to (GRA drops the entropy cut, ERA drops the graph). Bold marks
the best accuracy per row (Table I) or per dataset column (ablation block),
and in Table I also the lowest cost among the pruning arms ERA/GRA/GERA
(SRA and LSM sit outside the cost bolding: SRA is near-free by construction
and would sweep it).

Writes results/paper_figs/gera/table1.tex and ablation_block.tex.

Usage:
    python3 -m scripts.shared.build_gera_tables
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "results" / "gera_suite.json"
OUT_DIR = REPO / "results" / "paper_figs" / "gera"

BUDGET = "16"
TABLE_MODELS = [
    ("llama31-8b", "Llama3.1-8B"),
    ("llama31-70b", "Llama3.1-70B"),
    ("qwen35-122b-100", "Qwen3.5-122B"),
]
DS_ORDER = [
    ("msmarco_criteria", "msmarco crit."),
    ("curated_criteria", "curated crit."),
    ("msmarco_pragma", "msmarco prag."),
    ("curated_pragma", "curated prag."),
]
ABLATION_MODEL = "qwen3-32b"
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
        f"\\caption{{Every search strategy against exhaustive BFS at $b{{=}}{budget}$"
        " for the budgeted methods. Bold marks the row's best accuracy and the"
        " lowest cost among the pruning arms (ERA, GRA, GERA).}",
        "\\label{tab:cost-savings}\\end{table*}",
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
    ("llama31-8b", "Llama3.1-8B"),
    ("llama31-70b", "Llama3.1-70B"),
    ("qwen3-32b", "Qwen3-32B"),
    ("qwen35-122b-100", "Qwen3.5-122B"),
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


if __name__ == "__main__":
    main()
