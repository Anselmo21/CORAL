# MRA: built-in vs pyflagr mct/rra (Llama-3.1-8B, msmarco_criteria)

- normalize_fn="softmax", n=156
- pyflagr methods collapse each estimated child's score vector to a ranking, run the classical aggregator, and re-normalize.

| Budget | Impl | Acc | Cost | Hop-miss μ |
|---:|---|---:|---:|---:|
| 4 | built-in | 97.4% | 75.0 | 4.00 |
| 4 | pyflagr mct | 98.7% | 83.6 | 4.00 |
| 4 | pyflagr rra | 95.5% | 94.8 | 2.60 |

| 8 | built-in | 96.8% | 74.5 | 3.67 |
| 8 | pyflagr mct | 98.7% | 85.5 | 4.00 |
| 8 | pyflagr rra | 95.5% | 98.7 | 2.60 |

| 16 | built-in | 96.8% | 73.6 | 2.33 |
| 16 | pyflagr mct | 98.1% | 91.5 | 4.00 |
| 16 | pyflagr rra | 95.5% | 104.5 | 2.60 |
