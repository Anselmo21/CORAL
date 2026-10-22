"""Backends that send prompts to an LLM through vLLM or an OpenAI-compatible API"""

from __future__ import annotations

from coral.llm.language_model import Completion, LanguageModel
from coral.llm.language_model_factory import LanguageModelFactory

__all__ = [
    'Completion',
    'LanguageModel',
    'LanguageModelFactory',
]
