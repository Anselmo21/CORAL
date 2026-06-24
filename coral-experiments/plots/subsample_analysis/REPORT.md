# Subsample lattice audit — MSMARCO & Sonnet

## What was analyzed

The project docs were recently rewritten to a "full-set / no dev–test split" methodology, but the new `data/{msmarco,sonnet}_subsample/` directories don't yet exist. The four 148-case files matching the description live under `data/archive/`:

- `data/archive/msmarco_subsample/`
- `data/archive/msmarco_test_subsample/`
- `data/archive/sonnet_subsample/`
- `data/archive/sonnet_test_subsample/`

All four use `target_per_size = 20` for k ∈ 3..9 plus 8 full k=10 lattices (one per qid) = **148 cases each, 592 total**.

---

## 1. Where the counterfactuals sit

CF layer = number of criteria in the BFS-found CF subset. Most CFs sit in the middle layers of the start node, but distributions differ:

| Set | layer 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| msmarco_dev  | 0  | 13 | 29 | 25 | **52** | 18 | 5  | 6 |
| sonnet_dev   | 0  | **34** | **38** | 31 | 29 | 6  | 3  | 7 |
| msmarco_test | **10** | 21 | 16 | 25 | **30** | 26 | 14 | 6 |
| sonnet_test  | 0  | **30** | **34** | **33** | 25 | 10 | 11 | 5 |

- **Sonnet skews shallow** (CF at layers 1–3 dominate) — the full-lattice top-1 changes quickly as criteria are removed; many sub-lattices flip near the top.
- **MSMARCO skews deeper** — modal layer is 4. The model is more "sticky" so you need to drop more criteria to flip.
- **CF depth from start node** (how many hops down from the start): mode is depth 1–3 across all four sets; the flip typically happens within a few removals.

Full `(start size k, CF layer)` matrix: see `cf_layer_heatmaps.png`.

---

## 2. Repeats

- **Exact `(qid, start_node)` duplicates: 0** in every set. Sampling is clean — no double-counted sub-lattices.
- **No cross-set leakage** — dev and test use disjoint qids in both datasets.
- **CF-target repetition is high.** Distinct `(qid, cf_key)` tuples vs total cases:

| Set | unique CFs / cases | rate |
|---|---|---|
| msmarco_dev  | 107 / 148 | 72.3% |
| msmarco_test |  98 / 148 | 66.2% |
| sonnet_dev   |  81 / 148 | 54.7% |
| sonnet_test  |  85 / 148 | 57.4% |

Worst single CFs:

- sonnet_test qid=54 (Tesla Model 3 federal tax credit eligibility) — rediscovered by **11** different sub-lattices.
- msmarco_test qid=441101 — rediscovered by **9** different sub-lattices.
- Many CFs repeat 5–8 times.

Many "different" cases collapse to the same answer-flip event. **Effective N is smaller than 148**, especially in Sonnet (~80–85 unique discoveries).

---

## 3. Sampling issues

1. **Per-qid imbalance (range 14–16 across qids).** With 8 qids and target 20-per-size, even spread would give ~17–19 cases per qid. Actual:
   - msmarco_dev:  11–25
   - msmarco_test: 10–25
   - sonnet_dev:   11–25
   - sonnet_test:  11–27 (sonnet_test qid=21 hits 27)

   The sampler doesn't stratify by qid.

2. **Per-(qid, size) hot cells reach 8** in msmarco_test and sonnet_test — a single qid contributes 8 of the 20 cases for some size. Couples performance numbers to a small handful of lattices.

3. **k=10 is undersampled by design** (8 vs 20). Standard, but worth flagging when reporting "accuracy by size".

4. **Layer-0 CFs only appear in msmarco_test (10 cases, all qid=441101).** The empty subset is reported as a CF — i.e. the model's no-criteria answer differs from the answer with the full start-node. Not a bug, but layer-0 is degenerate: no search method can "find" it by removing criteria — it's always the deepest possible target. 10/148 is meaningful skew driven by a single qid.

5. **CF reuse shrinks effective N.** Sonnet sets effectively have ~80–85 unique CF discoveries, not 148. Worth keeping in mind for confidence intervals on accuracy.

6. **Sub-lattice CFs at k=3 cluster at layers 1–2 only** — size-3 has only 7 non-trivial nodes, so "size=3 accuracy" is essentially a 1-or-2-layer problem.

7. **The archived files are stale** under the new methodology. The rewritten `run_subsample.py` pools all 16 qids per dataset (no dev/test split), so the per-qid skew likely shrinks — but each qid will still contribute a variable number of cases unless the sampler is changed to stratify.

---

## 4. Visualizations

All in `plots/subsample_analysis/`:

- **`cf_layer_heatmaps.png`** — `start size × CF layer` heatmap per dataset (4 panels).
- **`case_balance.png`** — per-qid case counts + per-(qid, size) heatmap.
- **`msmarco_dev_hasse_examples.png`**, **`msmarco_test_hasse_examples.png`**, **`sonnet_dev_hasse_examples.png`**, **`sonnet_test_hasse_examples.png`** — one Hasse diagram per size 3..10 per dataset. Nodes colored by top-1 answer; BFS CF marked with red ★.

Size-10 (1024 nodes) is dense; edges are suppressed above k=7 for legibility. The colored regions give an intuitive read on top-1 fragility: large monochrome regions = stable answer; many color patches = many counterfactual paths.

---

# Pragma datasets

## What was analyzed

Sourced from `data/msmarco-pragma-final8-with-lattices.json` (8 qids × 10 pragmas) and `data/sonnet-pragma-dataset.json` (16 qids × 10 pragmas). Archived sub-lattice files:

- `data/archive/msmarco_pragma/{cases_dev,cases_test,test_results}.json` — **144 dev / 143 test cases** (less than 148 because the CF must require ≥2 pragmas removed, plus only 4 full-k=10 lattices instead of 8).
- `data/archive/sonnet_pragma/{cases_dev,cases_test,test_results}.json` — **148 dev / 148 test cases** (8 full-k=10 lattices per split).

Pragmas are categorical labels like `(Temporal): 2023`, `(Geographic): USA`, `(Trustworthiness): user forum`, `(Perspective): formal degree advocate`. Each qid has 10 pragmas drawn from 4 categories.

## 1. CF locations

| Set | layer 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| msmarco_pragma_dev  | 14 | **59** | 10 | 12 | 22 | 12 | 11 | 3 | 1 |
| msmarco_pragma_test | 0  | **82** | 10 | 9  | 20 | 14 | 2  | 5 | 1 |
| sonnet_pragma_dev   | 0  | **42** | 18 | 23 | 20 | 24 | 17 | 3 | 1 |
| sonnet_pragma_test  | 0  | **46** | 14 | 33 | 18 | 15 | 11 | 11 | 0 |

- **Pragma CFs cluster heavily at layer 1.** A *single* pragma flip is typically enough to change top-1. That contrasts with criteria sets (modal layer 2–4). Pragmas are framing nuances — removing one often shifts the model.
- **k=3 sonnet pragma: 100% of CFs at layer 1** (both splits, 20/20). The ≥2-pragmas-removed constraint at k=3 only allows layers 0 or 1, and the sampler apparently filtered out layer-0 in sonnet.
- **k=3 msmarco_pragma_test: 100% of CFs at layer 1** as well. msmarco_pragma_dev does include 7/20 layer-0 CFs at k=3.
- **CF depth from start: mode is 2 across all four sets** (75 of 148 sonnet_pragma_test cases have CF exactly 2 hops below start).

See `pragma_cf_layer_heatmaps.png`.

## 2. Repeats

| Set | exact (qid,start_node) dups | unique CFs / cases | rate |
|---|---|---|---|
| msmarco_pragma_dev  | 0 | 47 / 144 | **32.6%** |
| msmarco_pragma_test | 0 | 36 / 143 | **25.2%** |
| sonnet_pragma_dev   | 0 | 94 / 148 | 63.5% |
| sonnet_pragma_test  | 0 | 85 / 148 | 57.4% |

- **MSMARCO pragma CF reuse is severe** — 25–32% unique rate means effective N is closer to 35–50, not 143–144. The same CF target gets rediscovered by many sub-lattices because the qid pool is tiny (3–4 qids).
- No cross-split overlap — disjoint qids in both datasets.

## 3. Sampling issues (pragma-specific)

1. **MSMARCO pragma has only 4 dev qids and 3 test qids.** Per-qid case counts:
   - msmarco_pragma_dev:  {441101: 33, 52329: 43, 51957: 22, 51894: 46}
   - msmarco_pragma_test: {51821: **79**, 51718: 45, 52040: 19}

   qid=51821 alone contributes **79 of 143 test cases (55%)** — any aggregate "MSMARCO-pragma test accuracy" number is essentially the score on one query.

2. **Sonnet pragma is more balanced** (8 qids per split) but still ranges 9–32 (dev) and 10–27 (test).

3. **Pragma category mix at k=10 is skewed for sonnet** — Trustworthiness=30 vs Geographic=16 across qids. This is inherited from the source dataset: sonnet qids have more Trustworthiness pragmas per query than other categories. Worth flagging if you report results broken down by pragma category.

4. **Layer-0 CFs appear only in msmarco_pragma_dev (14 cases).** The "empty subset is CF" outcome doesn't appear in any other pragma set. Same caveat as for criteria sets: layer-0 is unreachable by search-by-removal.

5. **Effective N is much smaller than 144/143 for MSMARCO pragma** — combine the qid skew and CF-target reuse and you really have a 3-qid / 35-distinct-CF benchmark.

6. **Trivial size-3 cases in sonnet.** Sonnet pragma k=3 only ever produces layer-1 CFs (20/20). Size-3 accuracy on sonnet pragma is therefore a 1-hop search problem — methods will look uniformly good.

## 4. Pragma visualizations

In `plots/subsample_analysis/`:

- **`pragma_cf_layer_heatmaps.png`** — `start size × CF layer` heatmap per pragma dataset.
- **`pragma_case_balance.png`** — per-qid case counts + per-(qid, size) heatmap (the msmarco_pragma_test skew toward qid=51821 is very visible).
- **`{msmarco,sonnet}_pragma_{dev,test}_hasse_examples.png`** — one Hasse per size 3..10 per pragma set, distinct hues for top-1 answers, red ★ for BFS CF.
