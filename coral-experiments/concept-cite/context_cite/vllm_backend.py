"""
vLLM-backed variant of ContextCiter.

Why this exists
---------------
The original ``ContextCiter`` runs inference through HuggingFace ``transformers``.
``VLLMContextCiter`` is a drop-in subclass that runs inference through vLLM
instead, for users who already serve their LMs with vLLM or want vLLM's
throughput for the many teacher-forced forward passes ContextCite needs.

Math equivalence
----------------
ContextCite's per-token score in ``utils._compute_logit_probs`` is

    output = z_c − logsumexp_{j != c}(z_j)
           = log(p_c / (1 − p_c))   = logit(p_c)

so it depends only on the chosen token's probability p_c, not on the full
vocab distribution. vLLM's ``prompt_logprobs`` API always returns the chosen
token's exact logprob regardless of the top-N parameter, so we can reconstruct
the same per-token score as:

    logit_p = log p_c − log1p(−exp(log p_c))

Numerical differences between this backend and the HF backend are dominated by
model dtype (vLLM commonly defaults to fp16/bf16 while HF often runs fp32),
not by the extraction logic.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .context_citer import ContextCiter
from .utils import _create_mask

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError(
        "vLLM is required for VLLMContextCiter. "
        "Install with `pip install context-cite[vllm]` or `pip install vllm`."
    ) from e

try:
    from vllm.inputs import TokensPrompt
except ImportError:
    try:
        from vllm import TokensPrompt  # type: ignore
    except ImportError:
        TokensPrompt = None  # type: ignore


DEFAULT_VLLM_GENERATE_KWARGS = {"max_tokens": 512, "temperature": 0.0}


def _hf_to_vllm_gen_kwargs(generate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Translate common HF generate kwargs to vLLM SamplingParams kwargs."""
    out: Dict[str, Any] = {}
    if "max_new_tokens" in generate_kwargs:
        out["max_tokens"] = generate_kwargs["max_new_tokens"]
    if "max_tokens" in generate_kwargs:
        out["max_tokens"] = generate_kwargs["max_tokens"]
    if generate_kwargs.get("do_sample") is False:
        out["temperature"] = 0.0
    if "temperature" in generate_kwargs:
        out["temperature"] = generate_kwargs["temperature"]
    if "top_p" in generate_kwargs:
        out["top_p"] = generate_kwargs["top_p"]
    if "top_k" in generate_kwargs:
        out["top_k"] = generate_kwargs["top_k"]
    if "stop" in generate_kwargs:
        out["stop"] = generate_kwargs["stop"]
    if "seed" in generate_kwargs:
        out["seed"] = generate_kwargs["seed"]
    out.setdefault("max_tokens", DEFAULT_VLLM_GENERATE_KWARGS["max_tokens"])
    out.setdefault("temperature", DEFAULT_VLLM_GENERATE_KWARGS["temperature"])
    return out


def _wrap_token_ids(token_ids_list: List[List[int]]):
    """Wrap a list of token-id sequences as inputs vLLM.generate accepts."""
    if TokensPrompt is not None:
        return [TokensPrompt(prompt_token_ids=ids) for ids in token_ids_list]
    return [{"prompt_token_ids": ids} for ids in token_ids_list]


def _extract_response_log_probs(
    output, prompt_len: int, response_ids: List[int]
) -> NDArray:
    """Pull the chosen-token logprob for each response position."""
    log_probs = np.zeros(len(response_ids), dtype=np.float64)
    for j, tok in enumerate(response_ids):
        pos = prompt_len + j
        entry = output.prompt_logprobs[pos]
        if entry is None or tok not in entry:
            raise RuntimeError(
                f"vLLM did not return a logprob for token id {tok} at "
                f"prompt position {pos}; got: {entry!r}"
            )
        lp = entry[tok]
        log_probs[j] = lp.logprob if hasattr(lp, "logprob") else float(lp)
    return log_probs


def _vllm_teacher_forced_logit_probs(
    llm: "LLM",
    masks: NDArray,
    get_prompt_ids,
    response_ids,
    show_progress: bool = True,
    usage=None,
) -> NDArray:
    """Batched teacher-forced logit-prob computation via vLLM prompt_logprobs."""
    response_ids = list(response_ids)
    token_id_lists: List[List[int]] = []
    prompt_lens: List[int] = []
    for i in range(len(masks)):
        prompt_ids = list(get_prompt_ids(mask=masks[i]))
        prompt_lens.append(len(prompt_ids))
        token_id_lists.append(prompt_ids + response_ids)

    sp = SamplingParams(
        # temperature=1.0,
        temperature=0.0,
        seed=0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=1,
    )
    outputs = llm.generate(
        _wrap_token_ids(token_id_lists),
        sampling_params=sp,
        use_tqdm=show_progress,
    )
    if usage is not None:
        usage["prompt_tokens"] += sum(len(t) for t in token_id_lists)
        usage["output_tokens"] += sum(len(o.outputs[0].token_ids) for o in outputs)
        usage["llm_calls"] += len(token_id_lists)

    logit_probs = np.zeros((len(masks), len(response_ids)), dtype=np.float32)
    for i, out in enumerate(outputs):
        log_p = _extract_response_log_probs(out, prompt_lens[i], response_ids)
        # logit(p) = log p − log(1 − p); log1p for numerical stability
        # When log_p is very close to 0 (p ≈ 1), this saturates to +inf,
        # matching the HF path's behavior in the same regime.
        logit_p = log_p - np.log1p(-np.exp(log_p))
        logit_probs[i] = logit_p.astype(np.float32)
    return logit_probs


class VLLMContextCiter(ContextCiter):
    """
    Drop-in subclass of :class:`~context_cite.ContextCiter` that runs inference
    through vLLM instead of HuggingFace.

    The first positional argument is a ``vllm.LLM`` instance (kept on the
    ``model`` attribute for parity with the parent class; also aliased as
    ``self.llm``). Everything else — partitioners, solver, aggregation,
    attribution — is inherited unchanged.
    """

    def __init__(
        self,
        llm: "LLM",
        tokenizer: Any,
        context: str,
        query: str,
        **kwargs,
    ) -> None:
        super().__init__(llm, tokenizer, context, query, **kwargs)
        self.llm = llm

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        context: str,
        query: str,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "VLLMContextCiter":
        """
        Load a VLLMContextCiter from a pretrained model via ``vllm.LLM``.

        Arguments:
            pretrained_model_name_or_path (str):
                Model name or path passed to ``vllm.LLM(model=...)``.
            context (str): see :meth:`ContextCiter.from_pretrained`.
            query (str): see :meth:`ContextCiter.from_pretrained`.
            llm_kwargs (Optional[Dict[str, Any]]):
                Extra kwargs forwarded to ``vllm.LLM`` (e.g.,
                ``dtype``, ``gpu_memory_utilization``, ``tensor_parallel_size``).
            **kwargs:
                Forwarded to :class:`ContextCiter`.
        """
        llm_kwargs = llm_kwargs or {}
        llm = LLM(model=pretrained_model_name_or_path, **llm_kwargs)
        tokenizer = llm.get_tokenizer()
        tokenizer.padding_side = "left"
        return cls(llm, tokenizer, context, query, **kwargs)

    @property
    def _output(self):
        if self._cache.get("output") is None:
            prompt_ids, prompt = self._get_prompt_ids(return_prompt=True)
            sp = SamplingParams(**_hf_to_vllm_gen_kwargs(self.generate_kwargs))
            results = self.llm.generate(
                _wrap_token_ids([list(prompt_ids)]),
                sampling_params=sp,
                use_tqdm=False,
            )
            gen_text = results[0].outputs[0].text
            self.usage["prompt_tokens"] += len(prompt_ids)
            self.usage["output_tokens"] += len(results[0].outputs[0].token_ids)
            self.usage["llm_calls"] += 1
            self._cache["output"] = prompt + gen_text
        return self._cache["output"]

    def _compute_masks_and_logit_probs(self) -> None:
        n = self.num_sources
        masks = np.zeros((self.num_ablations, n), dtype=bool)
        for s in range(self.num_ablations):
            masks[s] = _create_mask(n, self.ablation_keep_prob, s)
        logit_probs = _vllm_teacher_forced_logit_probs(
            self.llm,
            masks,
            self._get_prompt_ids,
            self._response_ids,
            usage=self.usage,
        )
        self._cache["reg_masks"] = masks
        self._cache["reg_logit_probs"] = logit_probs

    def _get_reference_logit_probs(self, *, empty: bool):
        key = "ref_lp_empty_criteria" if empty else "ref_lp_full_criteria"
        if self._cache.get(key) is None:
            n = self.num_sources
            mask = (
                np.zeros((1, n), dtype=bool)
                if empty
                else np.ones((1, n), dtype=bool)
            )
            self._cache[key] = _vllm_teacher_forced_logit_probs(
                self.llm,
                mask,
                self._get_prompt_ids,
                self._response_ids,
                show_progress=False,
                usage=self.usage,
            )
        return self._cache[key]
