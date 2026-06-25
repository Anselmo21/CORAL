# Counterfactual search in answer-criteria lattices

Search algorithms that find **minimal counterfactual subsets** of criteria
(or pragmas) which flip an LLM's top-1 answer to a question. Compares an
oracle (BFS) against three proxies (SRA, ERA, ContextCite) across four
datasets — two question-criteria sets (MSMARCO, Sonnet) and two pragma sets
(MSMARCO, Sonnet).

## Quick start

```bash
# 1. Create the conda env
conda env create -f environment.yml
conda activate coral

# 2. Regenerate the paper's per-dataset table + plots from the cached data
python -m scripts.shared.paper_results
python -m scripts.shared.merge_cc_paper
python -m scripts.shared.plot_paper_results

# Output lands in final_plots/.
```

If you only want to skim the results without running anything, the paper
plots are already on disk under `final_plots/`.

## Repository layout

```
search/                  core algorithms (BFS, SRA, ERA, ContextCite)
metric.py                shared accuracy / hop-distance metric
scripts/                 per-dataset pipelines + the cross-dataset paper pipeline
data/                    raw lattice sources + processed outputs (see below)
final_plots/             the paper's headline plots
```

### `data/`

```
data/
├ sources/                          raw lattice JSONs (one per dataset)
│   ├ msmarco_criteria.json
│   ├ sonnet_criteria.json
│   ├ msmarco_pragma.json
│   └ sonnet_pragma.json
├ msmarco_criteria/                 processed: cases, results, cc_results, paper_results
├ sonnet_criteria/
├ msmarco_pragma/
└ sonnet_pragma/
```

Each `data/sources/<X>.json` is the raw LLM-grading output. The sibling
`data/<X>/` directory contains everything `scripts/` derives from it. The
existing `paper_results.json` in each processed dir is the single source of
truth for the headline table.

### `scripts/`

```
scripts/msmarco/                    MSMARCO criteria pipeline
scripts/sonnet/                     Sonnet criteria pipeline
scripts/pragma/msmarco_regen/       MSMARCO pragma pipeline
scripts/pragma/sonnet/              Sonnet pragma pipeline
scripts/shared/                     cross-dataset paper pipeline + helpers
```

Each per-dataset folder has the same shape:

| Script                       | Purpose                                              |
| ---------------------------- | ---------------------------------------------------- |
| `run_subsample.py`           | Sample sub-lattices, run BFS / SRA / ERA budget sweep |
| `plot_subsample.py`          | Per-dataset accuracy / cost vs k                     |
| `run_entropy_ablation.py`    | Compare 5 entropy variants at pct_10                 |
| `run_cc_full.py`             | ContextCite at one budget (b=16 or pct_10) — GPU     |
| `run_cc_ablation.py`         | ContextCite over the full budget grid — GPU          |
| `plot_cc_ablation.py`        | Per-dataset CC budget ablation plot                  |

### `scripts/shared/`

The paper pipeline — four steps that consume the four per-dataset
`results.json` files and produce `final_plots/`:

| Step | Script                       | What it does                                              |
| ---- | ---------------------------- | --------------------------------------------------------- |
| 1    | `paper_results.py`           | Refresh ERA at b=16 + pct_10, copy BFS/SRA, write paper_results.json |
| 2    | `run_cc_paper_parallel.sh`   | GPU launcher: 8 CC runs (4 datasets × 2 budgets) across N GPUs |
| 3    | `merge_cc_paper.py`          | Merge CC columns from disk into paper_results.json        |
| 4    | `plot_paper_results.py`      | Write final_plots/ — 4 per-dataset 3-panel figures + a cross-dataset headline |

Plus `plot_style.py`, `load_cc_model.py`, `cc_budget_ablation_dev.py`
(provides `score_cc()`).

## Running the full pipeline from scratch

```bash
# 1. Generate cases.json + results.json (BFS + SRA + ERA budget sweep) per dataset.
#    This is the slow step — minutes per dataset on CPU.
python -m scripts.msmarco.run_subsample
python -m scripts.sonnet.run_subsample
python -m scripts.pragma.msmarco_regen.run_subsample
python -m scripts.pragma.sonnet.run_subsample

# 2. Refresh the paper table with current ERA heuristics.
python -m scripts.shared.paper_results

# 3. Run ContextCite. Needs a GPU + vllm + transformers + a local context-cite
#    install (see environment.yml). Launcher handles the 8-runs-across-N-GPUs
#    coordination; check the script header for env vars + flags.
bash scripts/shared/run_cc_paper_parallel.sh

# 4. Merge CC columns + plot.
python -m scripts.shared.merge_cc_paper
python -m scripts.shared.plot_paper_results
```

## Headline results

ERA(b=16) Pareto-dominates the other proxies on every dataset:

| Dataset           | BFS cost | ERA(b=16) acc / cost | CC(pct_10) acc / cost |
| ----------------- | -------- | -------------------- | --------------------- |
| MSMARCO criteria  | 92.75    | 93.6% / 47.04        | 73.7% / 73.33         |
| Sonnet criteria   | 138.34   | 98.7% / 75.35        | 85.9% / 121.27        |
| MSMARCO pragma    | 95.96    | 94.9% / 63.69        | 90.0% / 102.13        |
| Sonnet pragma     | 80.52    | 96.8% / 62.83        | 92.3% / 92.74         |

See `final_plots/headline_acc_vs_cost.png` and `final_plots/<X>/combined.png`.
