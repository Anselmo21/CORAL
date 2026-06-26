# CORAL-data-gen

Generates base datasets and per-query criteria/pragma lattices for evaluating
LLM ranking under different filter combinations.

## Base datasets

Both files share the same schema — top-level
`{version, description, pragmas_model, criteria_model, rows: [...]}` with rows
of `query_id, query, sources (str), criteria [10], pragmas {temporal,
geographic, perspective, trustworthiness}, plausible_answers`.

| File | Source | Rows |
|---|---|---|
| `datasets/msmarco-data.json` | MSMARCO queries with Claude-Opus-regenerated pragmas; gpt-4o-mini criteria (33 rows) + Claude-authored criteria (20 rows) | 53 |
| `datasets/sonnet-dataset.json` | Synthetic dataset generated end-to-end by Claude Sonnet | 60 |

## Scripts

| Script | What it does | Input | Output |
|---|---|---|---|
| `generate_dataset_sonnet.py` | Generates the Sonnet base dataset (60 hardcoded query specs, one Claude API call per point) | env: `ANTHROPIC_API_KEY` | `datasets/sonnet-dataset.json` |
| `generate_msmarco_pragmas.py` | Fills/regenerates the `pragmas` field for each MSMARCO row via gpt-4o-mini | `datasets/msmarco-data.json` (or `--path`) | same file, updated in place |
| `generate_dataset.py` | Generates a single lattice: 2^N rankings for all combinations of N atoms. Multi-provider (OpenAI / vLLM / local HF). Called per-query by the runner | a config dict (or `config.json`) per query | `<output_dir>/<name>/{config.json, answers.json}` |
| `run_lattices.py` | Iterates a base dataset, calls `generate_dataset` per row. Resume-safe at row and combo levels | `--source` (any base dataset) | `lattices/<run-name>/<qid>/{config.json, answers.json}` |

Helpers used by `generate_dataset.py`: `combinatorics.py`, `ranking_utils.py`.

## Workflow

```
# Regenerate base datasets (only if needed)
python generate_dataset_sonnet.py
python generate_msmarco_pragmas.py

# Run lattices for a base dataset
python run_lattices.py \
    --source datasets/msmarco-data.json \
    --mode criteria \
    --run-name msmarco-criteria-openai \
    --provider openai --model gpt-4o-mini

python run_lattices.py \
    --source datasets/sonnet-dataset.json \
    --mode pragmas \
    --run-name sonnet-pragmas-vllm \
    --provider vllm --model Qwen/Qwen3-8B \
    --batch-size 32
```

`--mode criteria` uses each row's `criteria` field as the 10 atoms; `--mode
pragmas` flattens the pragmas dict into atoms.

Common flags: `--qids 51580,51587` (subset), `--limit N`, `--max-tokens 256`,
`--temperature 0`, `--force` (re-run completed rows), `--fill-missing` (resume
per-combo within a row).

## Halt behavior

`generate_dataset.py` short-circuits a lattice after the singleton layer if all
N singletons produced the same top-1 answer — the higher-order combos add
nothing in that case. The skipped state is recorded as `halted: true` in the
row's `config.json`.

## Output layout

```
lattices/                                  (gitignored)
└── <run-name>/
    └── <qid>/
        ├── config.json    name, question, criteria, mode, halted state, combos_total/combos_run
        └── answers.json   {combo_key: [ranked plausible_answers]}
```

`combo_key` is a comma-joined sorted list of active atoms; the empty-set entry
has key `""`.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install openai anthropic        # plus: vllm, transformers, torch, accelerate (for non-OpenAI providers)
export OPENAI_API_KEY=...                     # for OpenAI calls
export ANTHROPIC_API_KEY=...                  # for generate_dataset_sonnet.py
```
