"""
Compare HF (ContextCiter) and vLLM (VLLMContextCiter) attribution scores.

The HF formula reduces to logit(p_correct), so vLLM's chosen-token logprob is
sufficient to reconstruct it exactly. This test checks that the two backends
agree numerically when run on the same model in the same dtype, on the same
ablation masks, against the same teacher-forced response.

To eliminate first-token tie-breaking divergence in greedy decoding, we force
the vLLM instance to reuse the HF-generated response (so both backends score
the same target token sequence).

Run only one model at a time — vLLM holds the GPU for the whole process.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

vllm = pytest.importorskip("vllm")  # noqa: F401  (skip whole module if missing)
import torch as ch  # noqa: E402

from context_cite import ContextCiter  # noqa: E402
from context_cite.vllm_backend import VLLMContextCiter  # noqa: E402


MODEL_NAME = os.environ.get(
    "CC_VLLM_TEST_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was completed in 1889 as the entrance to the 1889 World's Fair. "
    "The tower is 330 metres tall and was the tallest man-made structure "
    "in the world for 41 years."
)
QUERY = "How tall is the Eiffel Tower and when was it built?"

NUM_ABLATIONS = 32
KEEP_PROB = 0.5


def _make_hf() -> ContextCiter:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=ch.float32)
    model.eval().cuda()
    return ContextCiter(
        model,
        tok,
        CONTEXT,
        QUERY,
        num_ablations=NUM_ABLATIONS,
        ablation_keep_prob=KEEP_PROB,
        batch_size=2,
    )


def _make_vllm() -> VLLMContextCiter:
    llm = vllm.LLM(
        model=MODEL_NAME,
        dtype="float32",
        gpu_memory_utilization=0.5,
        enforce_eager=True,  # disable CUDA graphs for deterministic numerics
    )
    tok = llm.get_tokenizer()
    tok.padding_side = "left"
    return VLLMContextCiter(
        llm,
        tok,
        CONTEXT,
        QUERY,
        num_ablations=NUM_ABLATIONS,
        ablation_keep_prob=KEEP_PROB,
    )


def _sync_response(src: ContextCiter, dst: ContextCiter) -> None:
    """Force ``dst`` to score the same response tokens as ``src``."""
    _ = src.response
    dst._cache["output"] = src._cache["output"]


@pytest.mark.slow
def test_vllm_matches_hf_attributions():
    cc_hf = _make_hf()
    cc_v = _make_vllm()
    _sync_response(cc_hf, cc_v)

    # tokenizers are loaded from the same model -> response_ids identical
    assert list(cc_v._response_ids) == list(cc_hf._response_ids), (
        "response token ids differ; tokenizers do not match"
    )

    # mask construction is deterministic via seeded _create_mask
    np.testing.assert_array_equal(cc_hf._masks, cc_v._masks)

    # per-token logit_probs should be very close in fp32
    lp_hf = cc_hf._logit_probs
    lp_v = cc_v._logit_probs
    assert lp_hf.shape == lp_v.shape
    np.testing.assert_allclose(lp_hf, lp_v, rtol=1e-2, atol=5e-2)

    # final attribution vectors: tight value agreement + ranking agreement
    attr_hf = cc_hf.get_attributions()
    attr_v = cc_v.get_attributions()

    corr = np.corrcoef(attr_hf, attr_v)[0, 1]
    assert corr > 0.99, f"attribution correlation too low: {corr:.4f}"
    np.testing.assert_allclose(attr_hf, attr_v, atol=1e-2)

    # top-k ordering must agree (k = min(3, num_sources))
    k = min(3, cc_hf.num_sources)
    top_hf = set(np.argsort(attr_hf)[::-1][:k].tolist())
    top_v = set(np.argsort(attr_v)[::-1][:k].tolist())
    assert top_hf == top_v, f"top-{k} differ: HF={top_hf} vLLM={top_v}"


if __name__ == "__main__":
    # Run as a standalone script for ad-hoc benchmarking / debugging.
    cc_hf = _make_hf()
    cc_v = _make_vllm()
    _sync_response(cc_hf, cc_v)

    print(f"response: {cc_hf.response!r}")
    print(f"num_sources: {cc_hf.num_sources}, num_ablations: {NUM_ABLATIONS}")

    lp_hf = cc_hf._logit_probs
    lp_v = cc_v._logit_probs
    diff = np.abs(lp_hf - lp_v)
    print(
        f"per-token logit_prob diff:  "
        f"max={diff.max():.3e}  mean={diff.mean():.3e}  median={np.median(diff):.3e}"
    )

    attr_hf = cc_hf.get_attributions()
    attr_v = cc_v.get_attributions()
    print(f"HF   attributions: {attr_hf}")
    print(f"vLLM attributions: {attr_v}")
    print(f"Pearson r: {np.corrcoef(attr_hf, attr_v)[0, 1]:.6f}")
    print(f"max |Δattr|: {np.abs(attr_hf - attr_v).max():.3e}")
