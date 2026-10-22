"""Backend that runs the model in this process with vLLM

Constrained decoding can force every reply to match the expected answer format
"""

from __future__ import annotations

import logging
from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from coral.llm.language_model import Completion, LanguageModel, Message

logger = logging.getLogger(__name__)


class VllmModel(LanguageModel):
    """Loads the model once and answers each batch in one call"""

    PROVIDER = 'vllm'
    CONSTRAINED_DECODING = True

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
    ) -> None:
        """Load the model into vLLM with the given serving settings

        Args:
            model: The model id to load
            tensor_parallel_size: The GPUs per model
            max_model_len: The cap on prompt plus output tokens
            gpu_memory_utilization: The share of GPU memory vLLM may reserve
            max_num_seqs: The cap on concurrent sequences
            max_num_batched_tokens: The cap on tokens per engine step
            enforce_eager: Whether to skip CUDA graph capture
            disable_custom_all_reduce: Whether to use NCCL instead of vLLM's custom all-reduce
        """
        super().__init__(model)
        self._serving = {
            'tensor_parallel_size': tensor_parallel_size,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens,
            'enforce_eager': enforce_eager,
            'disable_custom_all_reduce': disable_custom_all_reduce,
        }
        # Unset options keep vLLM's defaults
        options: dict[str, Any] = {
            key: value
            for key, value in self._serving.items()
            if key != 'tensor_parallel_size' and value is not None and value is not False
        }
        logger.info('loading %s with %s', model, self._serving)
        self.llm = LLM(
            model=model,
            dtype='auto',
            enable_prefix_caching=True,
            tensor_parallel_size=tensor_parallel_size,
            **options,
        )

    @property
    def serving(self) -> dict[str, Any]:
        """Return the serving settings recorded in each lattice and graph file

        Returns:
            The settings by name
        """
        return dict(self._serving)

    def complete(
        self,
        conversations: list[list[Message]],
        temperature: float,
        max_tokens: int,
        regex: str | None = None,
    ) -> list[Completion]:
        """Answer every conversation in one batched call

        Args:
            conversations: The chat conversations to answer
            temperature: The sampling temperature
            max_tokens: The token cap per answer
            regex: The pattern answers must match, if any

        Returns:
            One completion per conversation in the same order
        """
        # The sampling settings Qwen3 recommends when thinking is off
        parameters = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            structured_outputs=StructuredOutputsParams(regex=regex) if regex else None,
        )
        # Hybrid reasoning models would otherwise write out their thinking before the answer
        outputs = self.llm.chat(
            conversations,
            sampling_params=parameters,
            use_tqdm=False,
            chat_template_kwargs={'enable_thinking': False},
        )
        return [
            Completion(
                text=(output.outputs[0].text or '').strip(),
                prompt_tokens=len(output.prompt_token_ids or []),
                output_tokens=len(output.outputs[0].token_ids or []),
                finish_reason=output.outputs[0].finish_reason,
            )
            for output in outputs
        ]
