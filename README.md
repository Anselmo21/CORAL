# CORAL

CORAL is a counterfactual-based explanation tool for LLM answers to open-ended questions. CORAL uncovers the unstated assumptions behind the LLM's answer (its implicit provenance) as concepts and finds the smallest sets of concepts that change the answer when removed. This tool was introduced in our paper "Efficient Concept-Based Explanations over Implicit Provenance for LLM Answers" and this repository houses the code and datasets described in it.

## Installation

Install CORAL's packages with Python 3.10 or newer from the repository root.

```bash
pip install -e .
pip install -e '.[aggregation]'
```

CORAL uses PyFlagr for the rank aggregation methods other than Borda. The second command installs PyFlagr. On Apple Silicon the flagr.dylib in PyFlagr is built for x86-64. Build an arm64 one from [FLAGR](https://github.com/lakritidis/FLAGR) with clang++ -O2 -std=c++11 -shared -fPIC cflagr.cpp -o flagr.dylib and copy the result into the installed pyflagr package.

Running strategies and reproducing the paper work on a laptop without GPUs or API keys because the data folder already holds each LLM's ranking of the answers under every concept subset.

## Running a strategy

The run_strategy command runs one search strategy on the test cases and prints its accuracy, cost and distance for each LLM and dataset. Accuracy is how often the strategy finds a minimal counterfactual, cost is its mean number of LLM calls per case, and distance is how many concepts its non-minimal results are off by.

```bash
python -m coral.run_strategy --strategy gera --budget 16 --entropy softmax --aggregator borda \
    --llms qwen3-32b --datasets curated --concept-sets criteria
```

The strategy is one of bfs, sra, era, gra, gera and got. The budget, entropy scorer and rank aggregator only apply to ERA, GRA and GERA. The test cases come from the LLMs (llama31-8b, llama31-70b, qwen3-32b, qwen35-122b), datasets (msmarco, curated) and concept sets (criteria, pragmas) you pass. Leaving one of these flags out runs all of its values. Bottom-up search adds concepts to the empty set instead of removing them from the full set. To search bottom-up, run python -m coral.sample_cases --direction bottom-up once, then add --direction bottom-up to run_strategy, evaluate_strategies or write_report.

## Reproducing the paper

```bash
python -m coral.evaluate_strategies
python -m coral.write_report
```

This writes Table I and Figures 4 and 5 to results/top-down. For Table II, run GERA under every entropy scorer and every rank aggregator (about 20 minutes, needs PyFlagr) and write the report again.

```bash
python -m coral.evaluate_strategies --configurations
python -m coral.write_report
```

LSM calls the LLM during its search and cannot replay the stored lattices. The run_strategy command leaves LSM out while evaluate_strategies scores the stored LSM runs in data/surrogate.

## Regenerating the data

This needs GPUs. We used vLLM 0.27.1 on four L40S GPUs at temperature 0. To match our runs, pass the vLLM flags saved under serving in each config.json and graph.json in data (all runs used --tensor-parallel-size 4).

```bash
pip install -e '.[generation,surrogate]'
python -m coral.generate_lattices --llm qwen3-32b --dataset msmarco --concept-set criteria
python -m coral.generate_graphs --llm qwen3-32b --dataset msmarco --concept-set criteria
python -m coral.sample_cases
python -m coral.fit_surrogates --llm qwen3-32b --dataset msmarco --concept-set criteria --budget 16
```

The LLMs are llama31-8b, llama31-70b, qwen3-32b and qwen35-122b, the datasets msmarco and curated, and the concept sets criteria and pragmas. To regenerate the curated questions with Claude, set ANTHROPIC_API_KEY and run coral.generate_dataset with a new --output file. The MS MARCO concepts were made outside this repository.

## Explaining your own question

```bash
pip install -e '.[api]'
python -m coral.propose_concepts "What is the best clarifying shampoo to remove color?" --output shampoo.json --provider anthropic --model claude-opus-5-5
python -m coral.explain_question shampoo.json --provider anthropic --model claude-opus-5-5
```

The first command asks the LLM for candidate answers and ten concepts and saves them to shampoo.json. Before the second, delete any concept you don't care about or move it to the pinned list to keep it in every prompt. The second command searches with GERA, asks the LLM only about the concept subsets it checks, and prints the concepts the answer depends on. The rankings are cached in a shampoo folder for later runs. The command takes the same strategy settings as run_strategy.

The example uses Claude. The --provider flag takes one of three backends.

- anthropic for Claude models with ANTHROPIC_API_KEY set
- openai for OpenAI chat models or any OpenAI-compatible server with OPENAI_API_KEY set or --base-url passed
- vllm for the four Hugging Face models behind data/lattices (meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-70B-Instruct, Qwen/Qwen3-32B, and Qwen/Qwen3.5-122B-A10B-FP8) on local GPUs after pip install -e '.[generation]'

## Acknowledgements

coral/surrogate adapts [ContextCite](https://github.com/MadryLab/context-cite) (MIT license in coral/surrogate/LICENSE).
