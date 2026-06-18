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


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    if args.vllm:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        from vllm import LLM
        print(f"Loading model via vLLM: {args.model} ...")
        model = LLM(model=args.model, dtype="float16")
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
