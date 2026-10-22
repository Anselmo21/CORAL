"""Backend for the OpenAI API and OpenAI-compatible local servers"""

from __future__ import annotations

import os

from openai import OpenAI

from coral.llm.language_model import Completion, LanguageModel, Message


class OpenAIModel(LanguageModel):
    """Sends each conversation as its own chat request"""

    PROVIDER = 'openai'
    CONSTRAINED_DECODING = False

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        """Create the client from the arguments or the environment

        Only a missing key with no base URL raises an error because a local server does not check
        the key

        Args:
            model: The model id to request
            base_url: The endpoint of an OpenAI-compatible server, or OPENAI_BASE_URL if None
            api_key: The API key, or OPENAI_API_KEY if None
        """
        super().__init__(model)
        base_url = base_url or os.getenv('OPENAI_BASE_URL')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            if not base_url:
                raise ValueError('set OPENAI_API_KEY or pass an API key')
            # Local servers ignore the key but the client needs one
            api_key = 'not-needed'
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        conversations: list[list[Message]],
        temperature: float,
        max_tokens: int,
        regex: str | None = None,
    ) -> list[Completion]:
        """Answer the conversations one request at a time

        Passing a regex raises a ValueError because the API cannot enforce a pattern

        Args:
            conversations: The chat conversations to answer
            temperature: The sampling temperature
            max_tokens: The token cap per answer
            regex: Must be None for this backend

        Returns:
            One completion per conversation in the same order
        """
        if regex is not None:
            raise ValueError('the OpenAI API cannot constrain answers to a pattern')
        completions = []
        for messages in conversations:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = response.usage
            completions.append(
                Completion(
                    text=response.choices[0].message.content or '',
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    finish_reason=response.choices[0].finish_reason,
                )
            )
        return completions
