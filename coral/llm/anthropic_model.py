"""Backend for Claude models through the Anthropic API"""

from __future__ import annotations

from anthropic import Anthropic

from coral.llm.language_model import Completion, LanguageModel, Message


class AnthropicModel(LanguageModel):
    """Sends each conversation to Claude as its own Messages request

    System messages move into the request's system field
    """

    PROVIDER = 'anthropic'
    CONSTRAINED_DECODING = False

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        """Create the client from the arguments or the environment

        Without an API key the client reads ANTHROPIC_API_KEY or a saved Anthropic login

        Args:
            model: The Claude model id
            base_url: The API endpoint, or None for the default one
            api_key: The API key, or None to read it from the environment
        """
        super().__init__(model)
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def complete(
        self,
        conversations: list[list[Message]],
        temperature: float,
        max_tokens: int,
        regex: str | None = None,
    ) -> list[Completion]:
        """Answer the conversations one request at a time

        The temperature goes unused because the API rejects it

        Args:
            conversations: The chat conversations to answer
            temperature: The sampling temperature the other backends use
            max_tokens: The token cap per answer
            regex: Must be None for this backend

        Returns:
            One completion per conversation in the same order
        """
        del temperature
        if regex is not None:
            raise ValueError('the Anthropic API cannot constrain answers to a pattern')
        completions = []
        for conversation in conversations:
            system = '\n\n'.join(
                message['content'] for message in conversation if message['role'] == 'system'
            )
            request = {
                'model': self.model,
                'max_tokens': max_tokens,
                'messages': [message for message in conversation if message['role'] != 'system'],
            }
            if system:
                request['system'] = system
            response = self.client.messages.create(**request)
            completions.append(
                Completion(
                    text=''.join(block.text for block in response.content if block.type == 'text'),
                    prompt_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    finish_reason=response.stop_reason,
                )
            )
        return completions
