"""Shared model-loading helper for all CC runner scripts.

Usage in a runner:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args)

With --vllm, loads the model via vllm.LLM (all ablations batched together —
much faster than HF transformers). Without --vllm, loads via HuggingFace
AutoModelForCausalLM. search/contextcite.py auto-detects which backend to use
based on the model type.
"""
from __future__ import annotations

import argparse
import os

# Must be set before torch/CUDA init (e.g. via contextcite import) and before vLLM starts.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--vllm", action="store_true",
        help="Use vLLM backend (faster; batches all ablations in one call)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="HF batch size for ablations (ignored with --vllm)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="GPUs to shard the vLLM model across (vllm backend only)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Cap the model's context length (vllm backend only). Models with "
             "huge default contexts (e.g. 262k) demand KV cache sized for one "
             "max-length request; CC prompts are a few thousand tokens, so "
             "16384 is ample and frees most of that reservation.",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="Cap concurrent sequences (vllm backend only). Hybrid/Mamba "
             "models need one Mamba cache block per sequence, and vLLM's "
             "default of 256 can exceed the available blocks. CC submits at "
             "most budget+1 prompts at once, so 64 is more than enough.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="Fraction of GPU memory vLLM may reserve (vllm backend only). "
             "Prompt-logprob scoring needs a few hundred MB of headroom per "
             "chunk, so keep this below vLLM's 0.90 default on <=48GB cards.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce", action="store_true",
        help="Use NCCL instead of vLLM's custom all-reduce kernel (vllm "
             "backend only). vLLM only drops that kernel above two PCIe-only "
             "GPUs, so a TP=2 group on cards without NVLink keeps it and then "
             "dies registering IPC buffers at the end of CUDA graph capture.",
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    if args.vllm:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        from vllm import LLM
        print(f"Loading model via vLLM: {args.model} ...")
        extra = {}
        if args.max_model_len is not None:
            extra["max_model_len"] = args.max_model_len
        if args.max_num_seqs is not None:
            extra["max_num_seqs"] = args.max_num_seqs
        if args.disable_custom_all_reduce:
            extra["disable_custom_all_reduce"] = True
        model = LLM(model=args.model, dtype="float16",
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    seed=0,
                    **extra)
        tokenizer = model.get_tokenizer()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model via HF transformers: {args.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto",
        )
    tokenizer.padding_side = "left"
    print("Model loaded.")
    return model, tokenizer
