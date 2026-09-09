# used to generate the datasets for the experiments
#
# Requirements:
#   pip install openai
#   Set OPENAI_API_KEY environment variable or pass api_key parameter

'''
example run using Qwen3-32B model with 2 GPUs:

export CUDA_VISIBLE_DEVICES=1,2
export HF_HOME=/disk/0/data0/katherine/hf_cache

python run_lattices.py \
    --source datasets/sonnet-dataset.json \
    --mode pragmas \
    --run-name sonnet-pragmas-qwen3-32b \
    --provider vllm \
    --model Qwen/Qwen3-32B \
    --temperature 0.7 \
    --tensor-parallel-size 2

'''

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from combinatorics import generate_combinations
from ranking_utils import combo_key_from_items


def create_llm_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    local_device: str = "auto",
    local_dtype: str = "auto",
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    disable_custom_all_reduce: bool = False,
    enforce_eager: bool = False,
) -> Any:
    """Build an LLM client for the chosen provider. Build once, reuse across queries.

    For provider='openai', pass base_url to hit any OpenAI-compatible endpoint
    (Ollama, LM Studio, llama.cpp server, vLLM --served). Falls back to
    OPENAI_BASE_URL env var, else the official OpenAI API. When base_url is set
    a real API key is usually unnecessary; a placeholder is used if none is given.

    max_model_len and gpu_memory_utilization are vllm-only; leaving them None
    keeps vLLM's own defaults.
    """
    if provider == "openai":
        if OpenAI is None:
            raise SystemExit("pip install openai")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            if base_url:
                api_key = "not-needed"  # local servers typically ignore auth
            else:
                raise ValueError("Set OPENAI_API_KEY or pass api_key=")
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    if provider == "vllm":
        try:
            from vllm import LLM
        except ImportError as exc:
            raise SystemExit("pip install vllm") from exc
        if not model:
            raise ValueError("model is required for provider='vllm'")
        # vLLM sizes the KV cache to hold one request at the model's full context.
        # Checkpoints advertising a 100k+ window leave nothing for it on smaller
        # cards, so cap max_model_len to what the prompts actually need.
        extra = {}
        if max_model_len is not None:
            extra["max_model_len"] = max_model_len
        if gpu_memory_utilization is not None:
            extra["gpu_memory_utilization"] = gpu_memory_utilization
        # caps the startup profile peak on GPUs shared with other processes
        if max_num_seqs is not None:
            extra["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            extra["max_num_batched_tokens"] = max_num_batched_tokens
        # TP=2 on PCIe-only GPUs crashes in vLLM's custom all-reduce
        if disable_custom_all_reduce:
            extra["disable_custom_all_reduce"] = True
        # skips CUDA graph capture; slower but avoids the capture-time spike
        if enforce_eager:
            extra["enforce_eager"] = True
        print(f"Loading vLLM model '{model}' (dtype={local_dtype}, "
              f"tensor_parallel_size={tensor_parallel_size}"
              + (f", max_model_len={max_model_len}" if max_model_len else "")
              + (f", gpu_memory_utilization={gpu_memory_utilization}" if gpu_memory_utilization else "")
              + ")...")
        client = LLM(
            model=model,
            dtype=local_dtype,
            enable_prefix_caching=True,
            tensor_parallel_size=tensor_parallel_size,
            **extra,
        )
        print("Model loaded.")
        return client
    if provider == "local":
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise SystemExit("pip install transformers torch accelerate") from exc
        if not model:
            raise ValueError("model is required for provider='local' (e.g. meta-llama/Llama-3.1-8B-Instruct)")
        print(f"Loading local model '{model}' (device_map={local_device}, torch_dtype={local_dtype})...")
        client = hf_pipeline(
            "text-generation",
            model=model,
            device_map=local_device,
            torch_dtype=local_dtype,
        )
        # Llama 3 (and other open chat models) ship without a pad token, which
        # blocks batched generation. Set pad → eos so batching works.
        try:
            tok = client.tokenizer
            if getattr(tok, "pad_token_id", None) is None:
                eos = getattr(tok, "eos_token_id", None) or getattr(client.model.config, "eos_token_id", None)
                if isinstance(eos, list):
                    eos = eos[0]
                if eos is not None:
                    tok.pad_token_id = eos
                    # Use left padding for decoder-only models so attention masks line up correctly.
                    tok.padding_side = "left"
                    print(f"  Set tokenizer.pad_token_id = {eos} (eos), padding_side='left' for batched gen.")
        except Exception as e:
            print(f"  Warning: could not set pad token automatically: {e}")
        print("Model loaded.")
        return client
    raise ValueError(f"Unknown provider: {provider}")


_USAGE: Dict[str, Any] = {"prompt_tokens": 0, "output_tokens": 0,
                          "calls": 0, "finish_reasons": []}


def reset_usage() -> None:
    """Zero the counters before a unit of work whose cost you want to record."""
    _USAGE.update(prompt_tokens=0, output_tokens=0, calls=0, finish_reasons=[])


def usage_so_far() -> Dict[str, Any]:
    """Totals across every _invoke_chat_completion since the last reset.

    Counts are None-safe: providers that do not report them leave the totals
    where they were, so a zero means "not reported", not "free".
    """
    return {"prompt_tokens": _USAGE["prompt_tokens"],
            "output_tokens": _USAGE["output_tokens"],
            "llm_calls": _USAGE["calls"],
            "finish_reasons": list(_USAGE["finish_reasons"])}


# (prompt_tokens, output_tokens) per output of the most recent invoke, in order.
_LAST_CALL_USAGES: List[tuple] = []


def _record_usage(prompt_tokens, output_tokens, finish_reason) -> None:
    _USAGE["calls"] += 1
    _USAGE["finish_reasons"].append(finish_reason)
    _LAST_CALL_USAGES.append((prompt_tokens, output_tokens))
    if prompt_tokens is not None:
        _USAGE["prompt_tokens"] += prompt_tokens
    if output_tokens is not None:
        _USAGE["output_tokens"] += output_tokens


def _invoke_chat_completion(
    *,
    client: Any,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    use_tqdm: bool = False,
) -> str:
    """Provider-agnostic chat completion. Returns the assistant message content as a string."""
    _LAST_CALL_USAGES.clear()
    if provider == "openai":
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        _record_usage(getattr(usage, "prompt_tokens", None),
                      getattr(usage, "completion_tokens", None),
                      resp.choices[0].finish_reason)
        return resp.choices[0].message.content or ""
    if provider == "vllm":
        from vllm import SamplingParams
        # Qwen3 non-thinking recommended sampling: top_p=0.8, top_k=20, min_p=0.
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )
        # Qwen3 (and other hybrid-reasoning models) emit <think> traces unless the
        # chat template is told to disable thinking. Unknown templates ignore this kwarg.
        outputs = client.chat(
            [messages],
            sampling_params=sp,
            use_tqdm=use_tqdm,
            chat_template_kwargs={"enable_thinking": False},
        )
        out = outputs[0]
        _record_usage(len(out.prompt_token_ids or []),
                      len(out.outputs[0].token_ids or []),
                      out.outputs[0].finish_reason)
        return (out.outputs[0].text or "").strip()
    if provider == "local":
        do_sample = temperature > 0
        result = client(
            messages,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        generated = result[0]["generated_text"]
        # transformers pipeline returns the conversation when given a messages list;
        # the last entry is the assistant turn.
        if isinstance(generated, list):
            return (generated[-1].get("content") or "").strip()
        return str(generated).strip()
    raise ValueError(f"Unknown provider: {provider}")


def _invoke_chat_completion_batch(
    *,
    client: Any,
    provider: str,
    model: str,
    messages_list: List[List[Dict[str, str]]],
    temperature: float,
    max_tokens: int,
    batch_size: int,
    use_tqdm: bool = False,
) -> List[str]:
    """Batched chat completion. For provider='local' uses transformers pipeline's native
    GPU batching. For provider='openai' falls back to sequential calls (no batching).
    Returns a list of response strings, same length and order as messages_list.
    """
    if not messages_list:
        return []
    _LAST_CALL_USAGES.clear()
    if provider == "vllm":
        from vllm import SamplingParams
        # Qwen3 non-thinking recommended sampling: top_p=0.8, top_k=20, min_p=0.
        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )
        outputs = client.chat(
            messages_list,
            sampling_params=sp,
            use_tqdm=use_tqdm,
            chat_template_kwargs={"enable_thinking": False},
        )
        for o in outputs:
            _record_usage(len(o.prompt_token_ids or []),
                          len(o.outputs[0].token_ids or []),
                          o.outputs[0].finish_reason)
        return [(o.outputs[0].text or "").strip() for o in outputs]
    if provider == "local":
        do_sample = temperature > 0
        results = client(
            messages_list,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
        local_texts: List[str] = []
        for r in results:
            # Pipeline output for batched chat input is per-input. Some versions wrap
            # the per-input result in a one-element list — normalize both cases.
            if isinstance(r, list):
                r = r[0] if r else {}
            generated = r.get("generated_text") if isinstance(r, dict) else None
            if isinstance(generated, list):
                local_texts.append((generated[-1].get("content") or "").strip())
            elif generated is not None:
                local_texts.append(str(generated).strip())
            else:
                local_texts.append("")
        return local_texts
    # OpenAI / fallback: sequential
    return [
        _invoke_chat_completion(
            client=client, provider=provider, model=model,
            messages=m, temperature=temperature, max_tokens=max_tokens,
        )
        for m in messages_list
    ]


def _iter_batches(
    to_process: List[tuple],
    provider: str,
    batch_size: int,
    fill_missing: bool,
):
    """
    Yield batches of (i, combo, combo_key) tuples.

    vLLM: two phases — empty-set + singletons first (so the halt check can fire
    between phases), then all higher-order combos in one shot. Each phase is handed
    to vLLM in a single client.chat call; vLLM's own continuous-batching scheduler
    decides concurrency from available GPU/KV-cache memory, so batch_size is ignored.
    fill_missing skips the two-phase split since there is no halt check.

    Other providers: fixed-size chunks of batch_size.
    """
    if provider == "vllm":
        if fill_missing:
            if to_process:
                yield to_process
        else:
            phase1 = [(i, c, k) for i, c, k in to_process if len(c) <= 1]
            phase2 = [(i, c, k) for i, c, k in to_process if len(c) > 1]
            if phase1:
                yield phase1
            if phase2:
                yield phase2
    else:
        bs = max(1, batch_size)
        idx = 0
        while idx < len(to_process):
            yield to_process[idx:idx + bs]
            idx += bs


def generate_all_criteria_combinations(criteria: List[str]) -> List[List[str]]:
    """
    Generate all combinations of criteria, including the empty set.
    
    Args:
        criteria: List of criteria strings
        
    Returns:
        List of all combinations (each is a list of criteria)
    """
    all_combos = []
    # Include empty set (size 0)
    all_combos.append([])
    # Include all sizes from 1 to len(criteria)
    for size in range(1, len(criteria) + 1):
        all_combos.extend(generate_combinations(criteria, size))
    return all_combos


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    inner = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    inner = inner.strip()
    if inner.startswith("json"):
        inner = inner[4:].strip()
    return inner


def create_full_ranking_prompt(
    question: str,
    criteria: List[str],
    sources: str,
    options: List[str],
) -> str:
    """Prompt that asks for a full permutation of OPTIONS as a JSON array.

    Format is intentionally identical to msmarco/extract_baseline_lattice.py so
    Llama-3.1-8B and other models see exactly the same instructions across the
    two pipelines.
    """
    criteria_text = (
        f"{', '.join(criteria)}."
        if criteria
        else "None specified."
    )
    n = len(options)
    options_block = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options))
    prompt = f"""
    You are an answering machine.
    Rank the options based on the following question using the following sources and question criteria, if provided.
    Weigh all criteria equally when ranking the options.
    The options are not provided in any order of importance, consider them all when answering the question.
    The answer ranked first is the best answer to the question and should be placed at the beginning of the list.

    Return a JSON array of exactly {n} integers — the 0-based indices of the options from OPTIONS, ordered best answer first.
    The array must be a permutation of the indices 0 to {n - 1}: include every index exactly once, with no duplicates and none missing.
    Output only the index numbers, not the option text.
    Do not include any explanation or additional text, just the JSON array.

    QUESTION:\n
    {question}

    SOURCES:\n
    {sources}

    CRITERIA:\n
    {criteria_text}

    OPTIONS (rank these {n} options by their index number):\n
    {options_block}

    ANSWER (JSON array of exactly {n} integers, a permutation of 0 to {n - 1}, best answer first):\n"""
    return prompt


def create_full_ranking_prompt_pragmas(
    question: str,
    pragmas: List[str],
    sources: str,
    options: List[str],
) -> str:
    """Pragma-mode ranking prompt.

    Frames pragmas as *source criteria* — filters on which sources to weight
    more heavily — rather than as answer-ranking criteria. Output format stays
    a verbatim JSON-array permutation of OPTIONS so downstream parsing
    (parse_full_ranking, bfs_eval, package_18_lattices) is unchanged.
    """
    prompt_pragma = " AND ".join(pragmas) if pragmas else "NO SOURCE CRITERIA"
    n = len(options)
    options_block = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options))
    prompt = f"""
    You are an answering machine.
    Rank the options to the following question using only the following sources.
    Heavily prioritize sources that match the provided source directives when answering the question.
    Source directives can be temporal (e.g., specific year or date range), geographic (e.g., a location or region), trustworthy (e.g., official, expert, user-generated), or perspective (e.g., neutral, opinionated).
    If no source directives are provided, all sources will be considered equally.
    The options are not provided in any order of importance, consider them all when answering the question.
    The answer ranked first is the best answer to the question and should be placed at the beginning of the list.

    Return a JSON array of exactly {n} integers — the 0-based indices of the options from OPTIONS, ordered best answer first.
    The array must be a permutation of the indices 0 to {n - 1}: include every index exactly once, with no duplicates and none missing.
    Output only the index numbers, not the option text.
    Do not include any explanation or additional text, just the JSON array.

    QUESTION:\n
    {question}

    SOURCES:\n
    {sources}

    SOURCE DIRECTIVES:\n
    {prompt_pragma}

    OPTIONS (rank these {n} options by their index number):\n
    {options_block}

    ANSWER (JSON array of exactly {n} integers, a permutation of 0 to {n - 1}, best answer first):\n"""
    return prompt


def build_retry_prompt(base_prompt: str, previous_response: str, reason: str) -> str:
    prev = (previous_response or "").strip()
    return (
        f"{base_prompt}\n\n"
        "Your previous answer was invalid.\n"
        f"Reason: {reason}\n"
        "Previous answer:\n"
        f"{prev}\n\n"
        "Now return a corrected answer that strictly follows all output rules.\n"
        "Return ONLY the JSON array."
    )


def _repair_json_array(response: str) -> str:
    """Strip fences and patch the common malformations chat models emit around
    JSON arrays (wrapping parens, a missing trailing bracket)."""
    repaired = _strip_code_fences(response).strip()
    if repaired.startswith("(") and repaired.endswith(")"):
        inner = repaired[1:-1].strip()
        if inner.startswith("["):
            repaired = inner
    if repaired.startswith("[") and not repaired.endswith("]"):
        repaired = repaired + "]"
    return repaired


def _as_index(x: Any) -> Optional[int]:
    """Coerce a JSON scalar to an int index, or None if it isn't one.
    Rejects bools (which are int subclasses) so True/False can't sneak in."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        return int(x.strip())
    return None


def parse_full_ranking(response: str, expected: List[str]) -> Optional[List[str]]:
    """Return a permutation of `expected` parsed from the model output, or None.

    Primary format is a JSON array of 0-based indices into `expected`
    (e.g. [2, 0, 1]). For robustness we also accept the legacy verbatim-string
    array. Either way the return value is a permutation of `expected` strings,
    so downstream consumers of answers.json are unchanged.
    """
    repaired = _repair_json_array(response)
    try:
        items = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None

    n = len(expected)

    # Primary: array of indices into `expected`.
    indices = [_as_index(x) for x in items]
    if all(idx is not None for idx in indices):
        if len(indices) == n and sorted(indices) == list(range(n)):
            return [expected[idx] for idx in indices]
        return None

    # Fallback: legacy verbatim-string permutation.
    ranked = [str(x).strip() for x in items]
    if len(ranked) != n or set(ranked) != set(expected):
        return None
    if len(ranked) != len(set(ranked)):
        return None
    return ranked


def _single_line_preview(text: str, max_chars: int = 400) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _parse_fail_reason(response: str, expected: List[str]) -> str:
    """Return a short human-readable reason why parse_full_ranking returned None."""
    repaired = _repair_json_array(response)
    try:
        items = json.loads(repaired)
    except json.JSONDecodeError as e:
        return f"json_error: {e}"
    if not isinstance(items, list):
        return f"not_a_list: got {type(items).__name__}"

    n = len(expected)

    # Index-format diagnostics (the format we now ask for).
    indices = [_as_index(x) for x in items]
    if all(idx is not None for idx in indices):
        if len(indices) != n:
            return f"wrong_count: got {len(indices)} indices, expected {n}"
        out_of_range = sorted({idx for idx in indices if idx < 0 or idx >= n})
        if out_of_range:
            return f"index_out_of_range: {out_of_range} (valid 0..{n - 1})"
        if len(set(indices)) != len(indices):
            return f"duplicate_indices: {sorted(indices)}"
        missing = sorted(set(range(n)) - set(indices))
        if missing:
            return f"missing_indices: {missing}"
        return "unknown"

    # Legacy verbatim-string diagnostics.
    ranked = [str(x).strip() for x in items]
    if len(ranked) != n:
        return f"wrong_count: got {len(ranked)}, expected {n}"
    extra = set(ranked) - set(expected)
    missing = set(expected) - set(ranked)
    if extra or missing:
        extra_preview = [_single_line_preview(s, 60) for s in extra]
        missing_preview = [_single_line_preview(s, 60) for s in missing]
        return f"string_mismatch: extra={extra_preview} missing={missing_preview}"
    if len(ranked) != len(set(ranked)):
        return "duplicates in output"
    return "unknown"


def generate_dataset(
    config_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    question: Optional[str] = None,
    criteria: Optional[List[str]] = None,
    plausible_answers: Optional[List[str]] = None,
    sources: Optional[str] = None,
    output_dir: str = "datasets",
    model: str = "gpt-4o-mini",
    mode: str = "criteria",
    provider: str = "openai",
    client: Any = None,
    local_device: str = "auto",
    local_dtype: str = "auto",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    serving: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    fill_missing: bool = False,
    batch_size: int = 1,
    use_tqdm: bool = False,
) -> Dict[str, List[str]]:
    """
    Generate a dataset by prompting an LLM for each criteria/pragma combination.

    Prompt depends on mode:
      - 'criteria': create_full_ranking_prompt — identical to msmarco/extract_baseline_lattice.py.
      - 'pragmas':  create_full_ranking_prompt_pragmas — frames atoms as "source criteria"
                    (filters on which sources to weight) rather than answer-ranking criteria.
    Both return a JSON-array permutation of OPTIONS so parsing is shared.

    Halts after the singleton layer if every singleton produced the same top-1 answer.

    Args:
        config_path: Path to existing config.json file (alternative to providing params directly)
        dataset_name: Name of the dataset
        question: The main question to answer
        criteria: List of criteria (or pragma atoms when mode='pragmas') to consider
        plausible_answers: Candidate answer strings to rank for this question. Required.
        sources: Source information/context for the LLM
        output_dir: Directory to save the dataset
        model: Model id — OpenAI name (gpt-4o-mini) or HF hub id (meta-llama/Llama-3.1-8B-Instruct)
        mode: 'criteria' or 'pragmas' — recorded in config.json for downstream tracking
        provider: 'openai' or 'local'
        client: Pre-built client; build once and reuse across many queries (important for local models)
        local_device / local_dtype: forwarded to transformers pipeline when provider='local'
        max_tokens, temperature: forwarded to the LLM
        api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        max_retries: Number of times to retry each combo on parse failure

    Returns:
        Dictionary mapping criteria combinations to ranked lists (each is a
        permutation of plausible_answers).
    """
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
        dataset_name = config.get("name", dataset_name)
        question = config.get("question", question)
        criteria = config.get("criteria", criteria)
        sources = config.get("sources", sources)
        plausible_answers = config.get("plausible_answers", plausible_answers)
        mode = config.get("mode", mode)

    if not all([dataset_name, question, criteria, sources, plausible_answers]):
        raise ValueError("Must provide either config_path or all of: dataset_name, question, criteria, plausible_answers, sources")
    if mode not in ("criteria", "pragmas"):
        raise ValueError(f"mode must be 'criteria' or 'pragmas', got {mode!r}")

    # Build the LLM client (or reuse one passed in — important for local Llama
    # so we don't reload the 8B weights for every query).
    if client is None:
        client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            local_device=local_device,
            local_dtype=local_dtype,
        )

    # Generate all combinations
    all_combos = generate_all_criteria_combinations(criteria)

    # If filling gaps in an existing run, load what's already done so we can
    # skip those combos and merge the new results.
    answers_dict: Dict[str, List[str]] = {}
    if fill_missing:
        existing_path = Path(output_dir) / dataset_name / "answers.json"
        if existing_path.exists():
            try:
                answers_dict = json.loads(existing_path.read_text())
                print(f"Loaded {len(answers_dict)} existing combos; will only call LLM for missing ones.")
            except Exception as e:
                print(f"Could not load existing answers.json ({e}); starting fresh.")

    missing_count = sum(
        1 for combo in all_combos
        if (combo_key_from_items(combo) if combo else "") not in answers_dict
    )
    if fill_missing:
        print(f"Lattice size: {len(all_combos)}; missing: {missing_count}; will run {missing_count} LLM calls.")
    else:
        print(f"Generating rankings for {len(all_combos)} {mode} combinations...")

    reset_usage()
    node_usage: Dict[str, List[int]] = {}
    singleton_top1s: List[str] = []
    halted_reason: Optional[str] = None

    system_msg = (
        "You are a helpful assistant that ranks items as JSON arrays. "
        "Follow formatting instructions exactly."
    )

    # Build the list of combos that actually need LLM calls (skip already-done ones in fill_missing).
    to_process: List[tuple] = []
    for i, combo in enumerate(all_combos, 1):
        combo_key = combo_key_from_items(combo) if combo else ""
        if fill_missing and combo_key in answers_dict:
            continue
        to_process.append((i, combo, combo_key))

    bs = max(1, batch_size)
    if bs > 1 and provider == "local":
        print(f"Batching enabled: batch_size={bs} (local pipeline)")
    elif provider == "vllm":
        print("vLLM: each phase submitted in one call; vLLM schedules its own continuous batching (batch_size is ignored)")

    done_count = 0
    for batch in _iter_batches(to_process, provider, bs, fill_missing):
        # Halt check: fires at the start of phase 2 (or any chunk whose first combo
        # has size > 1) once all singletons agreed on the same top-1 answer.
        if not fill_missing and halted_reason is None and batch:
            next_combo = batch[0][1]
            if (
                len(next_combo) > 1
                and len(singleton_top1s) == len(criteria)
                and len(set(singleton_top1s)) == 1
            ):
                halted_reason = (
                    f"all {len(criteria)} singletons produced the same top-1 answer: "
                    f"'{singleton_top1s[0]}'"
                )
                print(f"\n⚠ Halting lattice: {halted_reason}")
                print(f"  Skipping {len(to_process) - done_count} remaining combinations.")
                break

        # Build prompts and messages for the batch
        base_prompts: List[str] = []
        messages_list: List[List[Dict[str, str]]] = []
        prompt_builder = (
            create_full_ranking_prompt_pragmas if mode == "pragmas"
            else create_full_ranking_prompt
        )
        for (_, combo, _) in batch:
            bp = prompt_builder(question, combo, sources, plausible_answers)
            base_prompts.append(bp)
            messages_list.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": bp},
            ])

        if use_tqdm and provider == "vllm":
            print(f"vLLM inference: {len(messages_list)} prompts...")
        # Initial batched call
        try:
            response_texts = _invoke_chat_completion_batch(
                client=client,
                provider=provider,
                model=model,
                messages_list=messages_list,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=bs,
                use_tqdm=use_tqdm,
            )
            batch_usages = list(_LAST_CALL_USAGES)
        except Exception as e:
            print(f"  ⚠ batch call failed ({e}); falling back to sequential for this batch")
            response_texts = []
            batch_usages = []
            for m in messages_list:
                try:
                    response_texts.append(_invoke_chat_completion(
                        client=client, provider=provider, model=model,
                        messages=m, temperature=temperature, max_tokens=max_tokens,
                        use_tqdm=use_tqdm,
                    ))
                    batch_usages.append(_LAST_CALL_USAGES[0] if _LAST_CALL_USAGES else (None, None))
                except Exception as e2:
                    print(f"    sequential fallback also failed: {e2}")
                    response_texts.append("")
                    batch_usages.append((None, None))
        if len(batch_usages) < len(batch):
            # some providers record fewer entries than the batch; keep the zip aligned
            batch_usages += [(None, None)] * (len(batch) - len(batch_usages))

        # Parse each response; retry failures individually
        for (i, combo, combo_key), base_prompt, text, call_usage in zip(
                batch, base_prompts, response_texts, batch_usages):
            display_key = combo_key if combo_key else "(no criteria)"
            print(f"[{i}/{len(all_combos)}] '{display_key}'")

            parsed = parse_full_ranking(text, plausible_answers)
            last_text = text
            final_usage = call_usage
            attempts_used = 1
            while parsed is None and attempts_used < max_retries:
                attempts_used += 1
                try:
                    rp = build_retry_prompt(
                        base_prompt, last_text,
                        "Not a strict JSON array permutation of the option indices 0..N-1.",
                    )
                    last_text = _invoke_chat_completion(
                        client=client, provider=provider, model=model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": rp},
                        ],
                        temperature=temperature, max_tokens=max_tokens,
                        use_tqdm=use_tqdm,
                    )
                    if _LAST_CALL_USAGES:
                        final_usage = _LAST_CALL_USAGES[0]
                    parsed = parse_full_ranking(last_text, plausible_answers)
                    if parsed is None:
                        print(f"  ⚠ retry {attempts_used}/{max_retries} parse_fail")
                except Exception as e:
                    print(f"  ⚠ retry {attempts_used}/{max_retries} error: {e}")

            if parsed is not None:
                if attempts_used > 1:
                    print(f"  ✓ Ranked (after {attempts_used} attempts): {parsed[:3]}{' …' if len(parsed) > 3 else ''}")
                else:
                    print(f"  ✓ Ranked: {parsed[:3]}{' …' if len(parsed) > 3 else ''}")
                answers_dict[combo_key] = parsed
                # node price = the parseable attempt; retry waste stays in the query totals
                node_usage[combo_key] = list(final_usage)
                if len(combo) == 1:
                    singleton_top1s.append(parsed[0].strip().lower())
            else:
                reason = _parse_fail_reason(last_text, plausible_answers)
                print(f"  ✗ parse_fail after {max_retries} attempts [{reason}]. Last: "
                      f"{_single_line_preview(last_text)!r}")

        done_count += len(batch)

    # Save results
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save answers.json
    answers_path = os.path.join(dataset_dir, "answers.json")
    with open(answers_path, 'w') as f:
        json.dump(answers_dict, f, indent=2)
    print(f"\nSaved answers to {answers_path}")

    # Save/update config.json. In fill_missing mode, preserve halted state
    # from the original run (we don't re-evaluate it).
    out_config_path = os.path.join(dataset_dir, "config.json")
    halted_flag = halted_reason is not None
    halted_reason_out = halted_reason
    if fill_missing and Path(out_config_path).exists():
        try:
            prev = json.loads(Path(out_config_path).read_text())
            halted_flag = bool(prev.get("halted", halted_flag))
            halted_reason_out = prev.get("halted_reason", halted_reason_out)
        except Exception:
            pass
    usage = usage_so_far()
    reasons: Dict[str, int] = {}
    for r in usage.pop("finish_reasons"):
        reasons[str(r)] = reasons.get(str(r), 0) + 1
    usage["finish_reasons"] = reasons
    if fill_missing and Path(out_config_path).exists():
        # Cumulative across the original run and any fills.
        try:
            prev_usage = json.loads(Path(out_config_path).read_text()).get("usage") or {}
            for k in ("prompt_tokens", "output_tokens", "llm_calls"):
                usage[k] += int(prev_usage.get(k) or 0)
            for r, n in (prev_usage.get("finish_reasons") or {}).items():
                usage["finish_reasons"][r] = usage["finish_reasons"].get(r, 0) + int(n)
        except Exception:
            pass
    config_out = {
        "name": dataset_name,
        "question": question,
        "mode": mode,
        "criteria": criteria,
        "plausible_answers": plausible_answers,
        "sources": sources,
        "halted": halted_flag,
        "halted_reason": halted_reason_out,
        "combos_total": len(all_combos),
        "combos_run": len(answers_dict),
        "usage": usage,
        "generation": {
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "serving": serving or {},
        },
    }
    with open(out_config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f"Saved config to {out_config_path}")

    # per-node prices: {combo_key: [prompt_tokens, output_tokens]}
    tokens_path = os.path.join(dataset_dir, "tokens.json")
    if fill_missing and Path(tokens_path).exists():
        try:
            prev_tokens = json.loads(Path(tokens_path).read_text())
            prev_tokens.update(node_usage)
            node_usage = prev_tokens
        except Exception:
            pass
    with open(tokens_path, 'w') as f:
        json.dump(node_usage, f)
    print(f"Saved per-node tokens to {tokens_path}")

    return answers_dict


if __name__ == "__main__":
    # Example usage
    import sys
    
    sources = """
    SOURCE 1
    Metrolink Light Rail: The Commuter Gold Standard [City Transit Authority / Urban Mobility Report 2026]:
    The Metrolink Light Rail remains the gold standard for high-density accessibility, providing direct core-access for thousands of passengers. It features a high-efficiency electric drivetrain that delivers up to 18 hours of continuous operation on a single grid-charge. The New Gen cars feature 10-car configurations making them 2x faster for peak-hour clearance than older bus models. Its AI-driven dispatch system is 3x faster at route optimization than previous manual signaling, supporting 'Smart Hub' features like real-time arrival sync and dynamic load balancing. However, the fixed-rail architecture means route expansions are permanent and require significant lead time, offering limited flexibility compared to road-based transit.

    SOURCE 2
    Solar-Bus BRT: The Sustainable Versatility Standard [Green Mobility Blog / TechTransit 2026]:
    The Solar-Bus Rapid Transit (BRT) is a premier 'Sustainable' fleet, powered by Snapdragon-integrated solar arrays on every roof. Its standout feature is a dedicated modular chassis, outperforming the Light Rail in long-term longevity and repairability due to its interchangeable parts and 45-minute battery swap capability. It matches the Light Rail's efficiency with up to 22 hours of peak operational range using local regenerative braking. It features a 120Hz digital route-mapping interface on every window, though solar-reliant fleets can still face occasional operational issues in niche northern climates during winter months.

    SOURCE 3
    Community Bike-Share: The Budget Powerhouse [Municipal Finance / Citizen Transport 2026]:
    For cities on a tight budget, the Community Bike-Share program offers a wide-reaching network of 16 docking stations for as low as $699 per unit investment. While the frame build quality uses more recycled aluminum than premium carbon rivals, it provides a 'marathon range' through a purely mechanical or hybrid pedal-assist system. It is marketed as a reliable, mid-range workhorse for municipalities that prioritize decentralized cost-efficiency and immediate deployment over high-end aesthetic infrastructure projects.
    """

    generate_dataset(
        dataset_name="transit",
        question="What is the best transit system for a city?",
        criteria=[
            "accessibility",
            "longevity",
            "cost efficiency",
            "operational range",
            "deployment speed",
        ],
        plausible_answers=["Light Rail", "Solar-Bus BRT", "Community Bike-Share"],
        sources=sources,
        top_k=3,
        model="gpt-4o-mini",
        include_confidence=False,
    )
