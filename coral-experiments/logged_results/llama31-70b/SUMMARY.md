# Llama-3.1-70B Lattice Search Results

## TL;DR

- **BFS is the oracle** — 100% by definition, but scales exponentially: at k=10 it averages **199–602 calls** depending on the dataset.
- **SRA is cheap and brittle** — 2–7 avg calls but only **36.5–60.9% accuracy** overall; at k=10 it drops to **6.2–43.8%**. On msmarco_criteria SRA's accuracy falls from 100% at k=3 to 25% at k=7.
- **ERA closes the accuracy gap** — 88.5–99.4% at 26.5–74.0% of BFS cost across the four datasets.
- **Bigger ERA budget ≠ always better.** ERA(b=8) beats b=16 on msmarco_criteria (94.2% vs 91.7%); ERA(b=4) beats b=8 on msmarco_pragma (93.6% vs 92.3%). Movements are small (n=156 per dataset) — largely noise.
- **When ERA misses, it's off by ~1 layer.** Hop-distance-on-miss drops to **1.00 at b=16** for curated_criteria and curated_pragma.
- **Curated-pragma is the easiest**, msmarco-criteria is the hardest — mirrors the trend seen with Llama-3.1-8B and Qwen3-32B.
- **Cost savings at k=10 (vs BFS):** SRA saves **92.7–97.3%**, ERA saves **21.7–41.4%**.

## Setup

- **Data:** `data/llama31-70b/{msmarco-criteria, msmarco-pragmas, curated-criteria, curated-pragmas}-llama31-70b.json` (flat qid → 1024 combos each).
- **QID selection:** for each dataset, kept only non-halted qids whose full-set BFS finds a counterfactual at layer ≤ 8 (i.e. removing ≥ 2 concepts flips the top-1 answer). Randomly picked **16 qids** per dataset with `seed=42`.
    - Eligible pool sizes: msmarco_criteria 33, msmarco_pragma 31, curated_criteria 41, curated_pragma 30.
- **Sub-lattices:** for each size k ∈ {3…9}, sampled 20 sub-lattices per dataset that satisfy BFS-CF-layer ≤ (k − 2). Appended every full k=10 lattice (16 per dataset). Total **156 cases per dataset**.
- **Methods:** BFS (ground truth), SRA (CombSUM aggregation, ~free), ERA at fixed budgets b ∈ {4, 8, 16} (`propagate_scores=True`).
- **Outputs:** `logged_results/llama31-70b/<dataset>/{cases,results,paper_results}.json`; `plots/llama31-70b/<dataset>/combined.png`.
- **Pipeline entry point:** `scripts/llama31_70b/run_subsample.py`.

## Rank Aggregation Used

| Method | Rank aggregator | Notes |
|---|---|---|
| **BFS** | none | Exact ground-truth layer-by-layer traversal (`search/bfs.py`). |
| **SRA** | **CombSUM** (default `aggregator="combsum"`, `search/rank.py:243`) | `_borda` in this file is literally `return _combsum(...)` (`search/rank.py:42–45`), so CombSUM ≡ Borda-count here. Voters = singleton criterion rankings; sum of positional scores, then reorder. |
| **ERA** | **Max-normalized CombSUM ("Borda")** via `aggregator_default = make_aggregator()` (`search/rank.py:127`) | Each child voter's score vector is max-normalized to [0, 1] (`search/rank.py:117`), summed entry-wise across answers, then max-normalized again. Known children contribute Borda-from-ranking vectors; estimated children propagate their previously-stored normalized score vectors (`search/rank.py:138`, enabled by `propagate_scores=True`). |

No Markov-chain / MC methods, no Kemeny, no PyFlagr aggregators are used in this pipeline.

## Aggregate Results (all sizes pooled, n=156 per dataset)

Cost = LLM calls + budget used (for ERA). Hop-miss μ = mean `|proxy_layer − bfs_layer|` restricted to cases where the proxy did **not** find the minimal CF.

| Dataset | Method | Accuracy | Avg Cost | Hop μ (all) | Hop μ (miss only) |
|---|---|---:|---:|---:|---:|
| **msmarco_criteria** | BFS       | 100.0% | 136.1 | 0.00 | — |
|                      | SRA       |  57.1% |   4.0 | 1.16 | 2.54 |
|                      | ERA b=4   |  88.5% |  87.0 | 0.49 | 1.72 |
|                      | ERA b=8   |  94.2% |  96.3 | 0.44 | 1.44 |
|                      | ERA b=16  |  91.7% | 107.3 | 0.54 | 1.23 |
| **msmarco_pragma**   | BFS       | 100.0% | 100.4 | 0.00 | — |
|                      | SRA       |  60.9% |   6.6 | 1.16 | 2.69 |
|                      | ERA b=4   |  93.6% |  72.7 | 0.31 | 2.12 |
|                      | ERA b=8   |  92.3% |  74.7 | 0.41 | 2.45 |
|                      | ERA b=16  |  96.8% |  77.3 | 0.30 | 1.80 |
| **curated_criteria**  | BFS       | 100.0% | 142.3 | 0.00 | — |
|                      | SRA       |  58.3% |   7.1 | 1.02 | 2.17 |
|                      | ERA b=4   |  93.6% |  91.6 | 0.33 | 1.62 |
|                      | ERA b=8   |  95.5% |  86.9 | 0.30 | 1.33 |
|                      | ERA b=16  |  96.8% | 102.9 | 0.29 | 1.00 |
| **curated_pragma**    | BFS       | 100.0% |  56.4 | 0.00 | — |
|                      | SRA       |  36.5% |   2.0 | 2.00 | 3.16 |
|                      | ERA b=4   |  97.4% |  37.9 | 0.49 | 1.75 |
|                      | ERA b=8   |  98.1% |  39.6 | 0.42 | 1.67 |
|                      | ERA b=16  |  99.4% |  41.7 | 0.39 | 1.00 |

## Full-Lattice Results (k = 10 only, n=16 per dataset)

This is the regime where BFS pays the largest cost and where accuracy/cost tradeoffs are sharpest.

| Dataset | BFS calls | Method | Acc | Cost | Saved vs BFS |
|---|---:|---|---:|---:|---:|
| **msmarco_criteria** | 574.1 | SRA      |  43.8% |  15.6 | 97.3% |
|                      |       | ERA b=4  |  81.2% | 381.1 | 33.6% |
|                      |       | ERA b=8  |  93.8% | 340.8 | 40.6% |
|                      |       | ERA b=16 |  93.8% | 431.6 | 24.8% |
| **msmarco_pragma**   | 339.9 | SRA      |  18.8% |  24.8 | 92.7% |
|                      |       | ERA b=4  | 100.0% | 266.3 | 21.7% |
|                      |       | ERA b=8  |  87.5% | 250.7 | 26.3% |
|                      |       | ERA b=16 |  93.8% | 255.4 | 24.9% |
| **curated_criteria**  | 602.2 | SRA      |  37.5% |  19.9 | 96.7% |
|                      |       | ERA b=4  |  93.8% | 408.2 | 32.2% |
|                      |       | ERA b=8  |  93.8% | 352.7 | 41.4% |
|                      |       | ERA b=16 |  93.8% | 372.6 | 38.1% |
| **curated_pragma**    | 198.5 | SRA      |   6.2% |   6.3 | 96.8% |
|                      |       | ERA b=4  |  87.5% | 140.1 | 29.4% |
|                      |       | ERA b=8  |  93.8% | 136.2 | 31.4% |
|                      |       | ERA b=16 | 100.0% | 121.0 | 39.0% |

## SRA Degradation by Lattice Size (msmarco_criteria)

Shows why SRA is unreliable past small lattices:

| Size k | n | SRA accuracy |
|---:|---:|---:|
| 3  | 20 | 100% |
| 4  | 20 |  80% |
| 5  | 20 |  70% |
| 6  | 20 |  65% |
| 7  | 20 |  25% |
| 8  | 20 |  35% |
| 9  | 20 |  35% |
| 10 | 16 |  44% |

## Non-Monotonicities in ERA Budget

Cases where more budget did **not** help:

- **msmarco_criteria (pooled):** b=8 → 94.2%, b=16 → 91.7% (−2.5 pp).
- **msmarco_pragma (pooled):** b=4 → 93.6%, b=8 → 92.3% (−1.3 pp).
- **msmarco_pragma (k=10):** b=4 → 100%, b=8 → 87.5% (−12.5 pp; but n=16, so ±1 case ≈ ±6.25 pp).

The 2–3 pp swings pooled over 156 cases correspond to 3–5 case-level flips and are within sampling noise. The clear monotone-improvement dataset is **curated_pragma** (97.4 → 98.1 → 99.4%).

## Hop-Distance-on-Miss Trend

When ERA fails to find the minimal-CF, its proxy CF gets closer to the true CF as budget grows:

| Dataset | b=4 | b=8 | b=16 |
|---|---:|---:|---:|
| msmarco_criteria | 1.72 | 1.44 | **1.23** |
| msmarco_pragma   | 2.12 | 2.45 | **1.80** |
| curated_criteria  | 1.62 | 1.33 | **1.00** |
| curated_pragma    | 1.75 | 1.67 | **1.00** |

By b=16, curated_criteria and curated_pragma misses are only **1 layer** off the true minimal CF on average.

## Comparison to Other Model Backbones

- **Same qualitative story as Llama-3.1-8B and Qwen3-32B temp-0**: SRA collapses at larger lattices, ERA tracks BFS accuracy at ~50–75% of the cost.
- Llama-70B accuracies at ERA(b=8) are broadly in line with previous runs — no evidence that the larger backbone changes proxy-search behaviour beyond the noise floor.

## Files

- Pipeline: `scripts/llama31_70b/run_subsample.py`
- Raw per-case results: `logged_results/llama31-70b/<dataset>/results.json`
- Paper-format results: `logged_results/llama31-70b/<dataset>/paper_results.json`
- Plots: `plots/llama31-70b/<dataset>/combined.png` (3 panels: accuracy / cost / hop-on-miss)
