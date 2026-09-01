# MRA: Built-in Borda vs pyflagr Borda

- Normalization: `normalize_fn="softmax"` (both runs, matches `plot_qwen3_temp0_bfs_lsm.py`).
- Budgets: [4, 8, 16].
- Cases: reused from each model's existing `cases.json` (156 per dataset when full).

**Built-in Borda** = `search.rank._mra_aggregate_borda`: unnormalized Borda-count voter score vectors summed entry-wise across children.

**pyflagr Borda** = `pyflagr.Linear.BordaCount` via `search.flagr.make_era_aggregator("borda")`: each child voter's score vector is collapsed to a ranking, written to a CSV, aggregated by FLAGR's Borda, then max-normalized. Rank-only aggregator → positional scores.


## llama31-8b

### msmarco_criteria (n=156)

| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |
|---:|---|---:|---:|---:|---:|---:|
| 4 | built-in | 97.4% | 75.0 | 4.00 | 75.6% | 89.1% |
| 4 | pyflagr  | 90.4% | 75.8 | 1.92 |  |  |
| 8 | built-in | 96.8% | 74.5 | 3.67 | 73.1% | 89.1% |
| 8 | pyflagr  | 91.0% | 81.3 | 1.83 |  |  |
| 16 | built-in | 96.8% | 73.6 | 2.33 | 72.4% | 89.1% |
| 16 | pyflagr  | 91.7% | 85.2 | 1.91 |  |  |

### msmarco_pragma (n=156)

| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |
|---:|---|---:|---:|---:|---:|---:|
| 4 | built-in | 92.9% | 47.0 | 3.36 | 73.7% | 80.1% |
| 4 | pyflagr  | 85.9% | 51.9 | 1.91 |  |  |
| 8 | built-in | 94.9% | 48.1 | 3.00 | 78.8% | 84.6% |
| 8 | pyflagr  | 88.5% | 60.3 | 1.44 |  |  |
| 16 | built-in | 94.9% | 54.8 | 4.62 | 77.6% | 84.6% |
| 16 | pyflagr  | 87.2% | 69.7 | 1.30 |  |  |


## llama31-70b

### msmarco_criteria (n=156)

| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |
|---:|---|---:|---:|---:|---:|---:|
| 4 | built-in | 91.7% | 50.3 | 1.15 | 75.0% | 82.7% |
| 4 | pyflagr  | 84.0% | 64.2 | 2.92 |  |  |
| 8 | built-in | 94.9% | 58.9 | 1.00 | 73.1% | 82.7% |
| 8 | pyflagr  | 85.3% | 86.1 | 2.65 |  |  |
| 16 | built-in | 96.2% | 62.5 | 1.00 | 72.4% | 84.6% |
| 16 | pyflagr  | 87.2% | 99.2 | 2.35 |  |  |

### msmarco_pragma (n=156)

| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |
|---:|---|---:|---:|---:|---:|---:|
| 4 | built-in | 94.2% | 38.0 | 2.14 | 73.7% | 85.3% |
| 4 | pyflagr  | 87.2% | 65.1 | 2.44 |  |  |
| 8 | built-in | 94.2% | 43.7 | 3.14 | 74.4% | 82.7% |
| 8 | pyflagr  | 87.2% | 71.9 | 2.44 |  |  |
| 16 | built-in | 96.2% | 45.9 | 2.75 | 77.6% | 85.3% |
| 16 | pyflagr  | 87.8% | 69.6 | 2.24 |  |  |


## qwen3-32b

### msmarco_criteria (n=156)

| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |
|---:|---|---:|---:|---:|---:|---:|
| 4 | built-in | 99.4% | 62.0 | 4.00 | 78.8% | 87.8% |
| 4 | pyflagr  | 88.5% | 81.1 | 1.28 |  |  |
| 8 | built-in | 98.7% | 63.7 | 2.50 | 75.6% | 86.5% |
| 8 | pyflagr  | 87.8% | 77.2 | 1.32 |  |  |
| 16 | built-in | 98.1% | 64.2 | 1.00 | 78.2% | 90.4% |
| 16 | pyflagr  | 91.0% | 92.3 | 1.14 |  |  |

### msmarco_pragma (n=156)

| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |
|---:|---|---:|---:|---:|---:|---:|
| 4 | built-in | 96.8% | 50.2 | 1.50 | 72.4% | 87.8% |
| 4 | pyflagr  | 89.1% | 66.0 | 1.14 |  |  |
| 8 | built-in | 92.9% | 45.1 | 1.12 | 64.7% | 83.3% |
| 8 | pyflagr  | 88.5% | 66.1 | 1.27 |  |  |
| 16 | built-in | 97.4% | 47.8 | 2.00 | 66.7% | 89.7% |
| 16 | pyflagr  | 90.4% | 74.2 | 1.67 |  |  |
