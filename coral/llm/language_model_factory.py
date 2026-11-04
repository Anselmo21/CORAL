"""Builds the vLLM, OpenAI, or Anthropic backend named by a command's --provider flag"""

from __future__ import annotations

from typing import Any

from coral.llm.language_model import LanguageModel

PROVIDERS = ('vllm', 'openai', 'anthropic')


class LanguageModelFactory:
    """Builds the vllm, openai, or anthropic backend"""

    @staticmethod
    def create(provider: str, model: str, **options: Any) -> LanguageModel:
        """Return a backend for the provider and model

        An unknown provider raises a ValueError that lists the valid names

        Args:
            provider: One of 'vllm', 'openai', and 'anthropic'
            model: The model id to serve
            **options: The backend's own constructor options

        Returns:
            The loaded backend
        """
        # Backends are optional and load only when chosen
        if provider == 'vllm':
            from coral.llm.vllm_model import VllmModel

            return VllmModel(model, **options)
        if provider == 'openai':
            from coral.llm.openai_model import OpenAIModel

            return OpenAIModel(model, **options)
        if provider == 'anthropic':
            from coral.llm.anthropic_model import AnthropicModel

            return AnthropicModel(model, **options)
        raise ValueError(f'unknown provider {provider!r}, choose from {PROVIDERS}')
