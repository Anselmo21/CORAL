# MRA: built-in vs pyflagr comb methods (Llama-3.1-8B, msmarco_criteria)

- normalize_fn="softmax", n=30
- pyflagr methods collapse each estimated child's score vector to a ranking, run the classical aggregator, and re-normalize.

| Budget | Impl | Acc | Cost | Hop-miss μ |
|---:|---|---:|---:|---:|
| 4 | built-in | 96.7% | 72.0 | 3.00 |
| 4 | pyflagr borda | 90.0% | 73.7 | 1.67 |
| 4 | pyflagr combsum | 90.0% | 73.7 | 1.67 |
| 4 | pyflagr combmnz | 90.0% | 73.7 | 1.67 |

| 8 | built-in | 96.7% | 69.5 | 3.00 |
| 8 | pyflagr borda | 90.0% | 76.9 | 1.67 |
| 8 | pyflagr combsum | 90.0% | 76.9 | 1.67 |
| 8 | pyflagr combmnz | 90.0% | 76.9 | 1.67 |

| 16 | built-in | 96.7% | 79.3 | 3.00 |
| 16 | pyflagr borda | 90.0% | 85.9 | 1.67 |
| 16 | pyflagr combsum | 90.0% | 85.9 | 1.67 |
| 16 | pyflagr combmnz | 90.0% | 85.9 | 1.67 |
