# MRA: built-in vs pyflagr mct/mc4/rra/outranking/copeland/dibra (Llama-3.1-8B, msmarco_criteria)

- normalize_fn="softmax", n=30
- pyflagr methods collapse each estimated child's score vector to a ranking, run the classical aggregator, and re-normalize.

| Budget | Impl | Acc | Cost | Hop-miss μ |
|---:|---|---:|---:|---:|
| 4 | built-in | 96.7% | 72.0 | 3.00 |
| 4 | pyflagr mct | 100.0% | 75.3 | nan |
| 4 | pyflagr mc4 | 93.3% | 82.2 | 1.00 |
| 4 | pyflagr rra | 100.0% | 79.6 | nan |
| 4 | pyflagr outranking | 86.7% | 77.3 | 2.25 |
| 4 | pyflagr copeland | 93.3% | 82.5 | 1.00 |
| 4 | pyflagr dibra | 93.3% | 75.3 | 5.50 |

| 8 | built-in | 96.7% | 69.5 | 3.00 |
| 8 | pyflagr mct | 100.0% | 76.3 | nan |
| 8 | pyflagr mc4 | 90.0% | 85.8 | 2.00 |
| 8 | pyflagr rra | 100.0% | 83.3 | nan |
| 8 | pyflagr outranking | 90.0% | 76.1 | 1.67 |
| 8 | pyflagr copeland | 93.3% | 84.2 | 1.00 |
| 8 | pyflagr dibra | 93.3% | 77.5 | 3.50 |

| 16 | built-in | 96.7% | 79.3 | 3.00 |
| 16 | pyflagr mct | 96.7% | 82.7 | 4.00 |
| 16 | pyflagr mc4 | 90.0% | 91.3 | 2.00 |
| 16 | pyflagr rra | 100.0% | 88.8 | nan |
| 16 | pyflagr outranking | 90.0% | 73.6 | 2.00 |
| 16 | pyflagr copeland | 93.3% | 89.2 | 1.00 |
| 16 | pyflagr dibra | 90.0% | 83.4 | 3.33 |
