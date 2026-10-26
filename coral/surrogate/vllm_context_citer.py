"""ContextCite running on vLLM instead of Hugging Face transformers"""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from coral.surrogate.context_citer import ContextCiter


class VllmContextCiter(ContextCiter):
    """Scores every ablation in one batched vLLM call

    The base class runs its own forward passes, while this one reads vLLM's prompt log probabilities
    """

    @staticmethod
    def load_model(model_id: str, **serving: Any) -> tuple[Any, Any]:
        """Load a model with vLLM in half precision

        Args:
            model_id: The Hugging Face model id
            **serving: The vLLM serving options (None and False keep the default)

        Returns:
            The engine and its tokenizer
        """
        # Spawned workers avoid this process's CUDA state
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        options = {
            key: value for key, value in serving.items() if value is not None and value is not False
        }
        model = LLM(model=model_id, dtype='float16', seed=0, **options)
        tokenizer = model.get_tokenizer()
        tokenizer.padding_side = 'left'
        return model, tokenizer

    def _generate(self, prompt_ids: list[int]) -> str:
        """Generate the response to the full prompt

        Temperature zero stands in for the base class's greedy decoding

        Args:
            prompt_ids: The prompt's token ids

        Returns:
            The response text
        """
        parameters = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        output = self.model.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)], sampling_params=parameters, use_tqdm=False
        )[0].outputs[0]
        self.usage['prompt_tokens'] += len(prompt_ids)
        self.usage['output_tokens'] += len(output.token_ids)
        self.usage['llm_calls'] += 1
        return output.text

    def _score(self, masks: NDArray[np.bool_], response_ids: list[int]) -> NDArray:
        """Return each response token's logit probability under each mask

        The response goes in as part of the prompt because vLLM reports a log probability for each
        prompt token

        Args:
            masks: One mask per ablation
            response_ids: The response's token ids

        Returns:
            The logit probabilities, one row per mask and one column per response token
        """
        prompts = [self.prompt_ids(mask) for mask in masks]
        parameters = SamplingParams(
            temperature=0.0, seed=0, top_p=1.0, top_k=-1, max_tokens=1, prompt_logprobs=1
        )
        outputs = self.model.generate(
            [TokensPrompt(prompt_token_ids=prompt + response_ids) for prompt in prompts],
            sampling_params=parameters,
            use_tqdm=len(prompts) > 1,
        )
        self.usage['prompt_tokens'] += sum(len(prompt) + len(response_ids) for prompt in prompts)
        self.usage['output_tokens'] += sum(len(output.outputs[0].token_ids) for output in outputs)
        self.usage['llm_calls'] += len(prompts)
        scores = np.zeros((len(prompts), len(response_ids)), dtype=np.float32)
        for row, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
            log_probabilities = np.array(
                [
                    output.prompt_logprobs[len(prompt) + position][token].logprob
                    for position, token in enumerate(response_ids)
                ],
                dtype=np.float64,
            )
            # Convert each log probability to a logit, with log1p for precision near one
            scores[row] = (log_probabilities - np.log1p(-np.exp(log_probabilities))).astype(
                np.float32
            )
        return scores
