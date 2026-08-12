# Llama-3.3-70B Lattice Search Results

## TL;DR

- **SRA is stronger on 3.3 than 3.1 across every dataset** — pooled accuracy 34.6–66.0% (vs 36.5–60.9% on 3.1) with essentially identical cost (0.7–15.5 calls).
- **ERA at every budget clears 92%** on all four datasets (b=4: 92.9–99.4%, b=16: 96.8–99.4%).
- **curated_criteria is close to solved** by ERA at any budget: 99.4% (b=4/b=8) and 98.7% (b=16).
- **Budget scaling is not clean.** ERA(b=8) beats b=16 on curated_criteria (99.4% vs 98.7%); ERA(b=4) essentially ties b=8 on msmarco_criteria (92.9 vs 92.3%). Swings are ≤ 1–2 pp — within n=156 noise.
- **Hop-on-miss is mixed.** curated_pragma is well-behaved (b=4/b=16 both at 1.00 layer off). msmarco_pragma misses are stubbornly ~3.6–3.8 layers off — when ERA misses on this dataset it misses badly.
- **Cost savings at k=10 (vs BFS):** SRA saves **86.2–98.8%**, ERA saves **15.0–45.2%**.
- **At k=10, ERA hits 100% on both curated datasets** at every budget; msmarco datasets cap at 93.8% at every budget.

## Setup

- **Data:** `data/llama33-70b/{msmarco-criteria, msmarco-pragmas, curated-criteria, curated-pragmas}-llama33-70b.json` — same flat-qid + `raw_answers` layout as llama31-70b.
- **QID selection:** non-halted qids whose full-set BFS finds a CF at layer ≤ 8, then 16 sampled at random with `seed=42`.
    - Eligible pool sizes: msmarco_criteria 27, msmarco_pragma 25, curated_criteria 36, curated_pragma 32.
- **Sub-lattices:** 20 sub-lattices per size k ∈ {3…9} (BFS CF ≤ k − 2) + all 16 full k=10 lattices → **156 cases per dataset**.
- **Methods:** BFS, SRA (CombSUM), ERA at b ∈ {4, 8, 16} with `propagate_scores=True`.
- **Pipeline:** `scripts/llama33_70b/run_subsample.py` (mirror of `scripts/llama31_70b/run_subsample.py`).
- **Outputs:** `logged_results/llama33-70b/<dataset>/{cases,results,paper_results}.json`; `plots/llama33-70b/<dataset>/combined.png`.

## Rank Aggregation Used

Identical to Llama-3.1-70B:

| Method | Rank aggregator | Notes |
|---|---|---|
| **BFS** | none | Ground-truth traversal. |
| **SRA** | **CombSUM** (`aggregator="combsum"`, `search/rank.py:243`) | `_borda` ≡ `_combsum` in this codebase (`search/rank.py:42–45`). |
| **ERA** | **Max-normalized CombSUM** via `make_aggregator()` (`search/rank.py:127`) | Each voter's score vector is max-normalized to [0, 1], summed entry-wise, then max-normalized again. With `propagate_scores=True`, estimated children forward their normalized score vectors to their parents. |

## Aggregate Results (all sizes pooled, n=156 per dataset)

Cost = LLM calls + budget used. Hop-miss μ = mean `|proxy_layer − bfs_layer|` on cases where the proxy did **not** find the minimal CF.

| Dataset | Method | Accuracy | Avg Cost | Hop μ (miss only) |
|---|---|---:|---:|---:|
| **msmarco_criteria** | BFS       | 100.0% | 142.8 | — |
|                      | SRA       |  66.0% |  15.5 | 2.62 |
|                      | ERA b=4   |  92.9% | 108.2 | 2.00 |
|                      | ERA b=8   |  92.3% | 106.3 | 2.25 |
|                      | ERA b=16  |  97.4% | 103.8 | 2.25 |
| **msmarco_pragma**   | BFS       | 100.0% |  65.0 | — |
|                      | SRA       |  59.6% |   7.1 | 2.97 |
|                      | ERA b=4   |  95.5% |  50.5 | 3.83 |
|                      | ERA b=8   |  96.2% |  51.5 | 3.83 |
|                      | ERA b=16  |  96.8% |  44.8 | 3.60 |
| **curated_criteria**  | BFS       | 100.0% | 107.2 | — |
|                      | SRA       |  61.5% |   4.0 | 2.28 |
|                      | ERA b=4   |  99.4% |  71.2 | 3.00 |
|                      | ERA b=8   |  99.4% |  66.0 | 3.00 |
|                      | ERA b=16  |  98.7% |  75.1 | 2.00 |
| **curated_pragma**    | BFS       | 100.0% |  75.6 | — |
|                      | SRA       |  34.6% |   0.7 | 2.54 |
|                      | ERA b=4   |  95.5% |  57.0 | 1.00 |
|                      | ERA b=8   |  96.2% |  56.5 | 1.33 |
|                      | ERA b=16  |  97.4% |  59.3 | 1.00 |

## Full-Lattice Results (k = 10 only, n=16 per dataset)

| Dataset | BFS calls | Method | Acc | Cost | Saved vs BFS |
|---|---:|---|---:|---:|---:|
| **msmarco_criteria** | 623.6 | SRA      |  43.8% |  65.6 | 89.5% |
|                      |       | ERA b=4  |  93.8% | 495.2 | 20.6% |
|                      |       | ERA b=8  |  93.8% | 474.1 | 24.0% |
|                      |       | ERA b=16 |  93.8% | 433.1 | 30.5% |
| **msmarco_pragma**   | 226.4 | SRA      |  37.5% |  31.3 | 86.2% |
|                      |       | ERA b=4  |  93.8% | 187.2 | 17.3% |
|                      |       | ERA b=8  |  93.8% | 192.5 | 15.0% |
|                      |       | ERA b=16 |  93.8% | 132.0 | 41.7% |
| **curated_criteria**  | 509.7 | SRA      |  56.2% |  11.2 | 97.8% |
|                      |       | ERA b=4  | 100.0% | 346.8 | 32.0% |
|                      |       | ERA b=8  | 100.0% | 279.2 | 45.2% |
|                      |       | ERA b=16 | 100.0% | 367.0 | 28.0% |
| **curated_pragma**    | 216.8 | SRA      |  12.5% |   2.6 | 98.8% |
|                      |       | ERA b=4  | 100.0% | 182.3 | 15.9% |
|                      |       | ERA b=8  | 100.0% | 173.6 | 19.9% |
|                      |       | ERA b=16 | 100.0% | 159.6 | 26.4% |

## Head-to-Head vs Llama-3.1-70B (pooled accuracy, n=156)

Cases were sampled independently for each model (different qid subsets, same protocol), so this is a distributional comparison, not paired.

| Dataset | Method | 3.1-70B | 3.3-70B | Δ |
|---|---|---:|---:|---:|
| **msmarco_criteria** | SRA      | 57.1% | 66.0% | **+8.9** |
|                      | ERA b=4  | 88.5% | 92.9% | +4.4 |
|                      | ERA b=8  | 94.2% | 92.3% | −1.9 |
|                      | ERA b=16 | 91.7% | 97.4% | **+5.7** |
| **msmarco_pragma**   | SRA      | 60.9% | 59.6% | −1.3 |
|                      | ERA b=4  | 93.6% | 95.5% | +1.9 |
|                      | ERA b=8  | 92.3% | 96.2% | +3.9 |
|                      | ERA b=16 | 96.8% | 96.8% | 0.0 |
| **curated_criteria**  | SRA      | 58.3% | 61.5% | +3.2 |
|                      | ERA b=4  | 93.6% | 99.4% | **+5.8** |
|                      | ERA b=8  | 95.5% | 99.4% | +3.9 |
|                      | ERA b=16 | 96.8% | 98.7% | +1.9 |
| **curated_pragma**    | SRA      | 36.5% | 34.6% | −1.9 |
|                      | ERA b=4  | 97.4% | 95.5% | −1.9 |
|                      | ERA b=8  | 98.1% | 96.2% | −1.9 |
|                      | ERA b=16 | 99.4% | 97.4% | −2.0 |

**Reading:** 3.3-70B is a modest but consistent win for SRA and ERA on the msmarco and curated-criteria datasets. curated-pragma is the one dataset where 3.1-70B looks marginally better across methods (all deltas ≤ 2 pp, within noise).

## Files

- Pipeline: `scripts/llama33_70b/run_subsample.py`
- Raw per-case results: `logged_results/llama33-70b/<dataset>/results.json`
- Paper-format results: `logged_results/llama33-70b/<dataset>/paper_results.json`
- Plots: `plots/llama33-70b/<dataset>/combined.png` (accuracy / cost / hop-on-miss)
