"""The LLMs, datasets, and concept sets the commands run over

One combination of LLM, dataset, and concept set is called a cell throughout the code
"""

from __future__ import annotations

LANGUAGE_MODELS = {
    'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama31-70b': 'meta-llama/Llama-3.1-70B-Instruct',
    'qwen3-32b': 'Qwen/Qwen3-32B',
    'qwen35-122b': 'Qwen/Qwen3.5-122B-A10B-FP8',
}

LANGUAGE_MODEL_NAMES = {
    'llama31-8b': 'Llama3.1-8B',
    'llama31-70b': 'Llama3.1-70B',
    'qwen3-32b': 'Qwen3-32B',
    'qwen35-122b': 'Qwen3.5-122B',
}

DATASETS = ('msmarco', 'curated')

CONCEPT_SETS = ('criteria', 'pragmas')
