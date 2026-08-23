"""
Generate a Graph of Thoughts (GoT) per row of a base dataset, for either the
criteria set or the pragma set of that row.

Mirrors run_lattices.py conventions:
  - loads msmarco-data.json or sonnet-dataset.json
  - --mode chooses criteria vs. pragmas (used to seed Layer 3 of the graph)
  - reuses create_llm_client / _invoke_chat_completion from generate_dataset.py

Output layout:
  graphs/<run-name>/<qid>/graph.json
    { query_id, query, mode, atoms, raw, graph: {nodes, edges} }

Usage:
    # Hosted OpenAI:
    python generate_graphs.py --source datasets/msmarco-data.json \\
        --mode criteria --run-name msmarco-criteria-gpt4o-graphs \\
        --provider openai --model gpt-4o-mini

    # Local vLLM in server mode (`vllm serve Qwen/Qwen3-32B --port 8000`):
    python generate_graphs.py --source datasets/msmarco-data.json \\
        --mode criteria --run-name msmarco-criteria-local \\
        --provider openai --openai-base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-32B

    # Or load vLLM in-process (no server, one Python process owns the GPU):
    python generate_graphs.py --source datasets/msmarco-data.json \\
        --mode criteria --run-name msmarco-criteria-local \\
        --provider vllm --model Qwen/Qwen3-32B --tensor-parallel-size 1
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from generate_dataset import (
    _invoke_chat_completion,
    _strip_code_fences,
    create_llm_client,
)
from graph_ops import pick_answer_with_all_atoms
from run_lattices import flatten_pragmas, load_rows, parse_qid_filter, stringify_sources


GRAPHS_ROOT = Path("graphs")


def build_graph_of_thoughts_prompt(
    sources_text: str,
    question: str,
    mode: str = "criteria",
    seeded_atoms: Optional[List[str]] = None,
    seeded_answers: Optional[List[str]] = None,
    baseline_answer: Optional[str] = None,
) -> str:
    """
    Layered reasoning-graph prompt.

    mode="criteria": Layer 3 nodes are answer-evaluation criteria — the key points
    the LLM weighs when comparing candidate answers to the question.

    mode="pragmas": Layer 3 nodes are source directives — filters on which sources
    to prioritize when answering (temporal, geographic, trustworthiness, perspective).
    Answers are then supported by criteria that hold under those source filters.

    When seeded_atoms is provided, every atom must appear as a Layer 3 node.
    When seeded_answers is provided, every answer must appear as a Layer 4 node
    (this is the standard use — Layer 4 is not free-form, it draws from a fixed set
    of plausible answers so downstream graph analysis can key on labels).
    """
    if mode not in ("criteria", "pragmas"):
        raise ValueError(f"mode must be 'criteria' or 'pragmas', got {mode!r}")

    if mode == "criteria":
        layer3_desc = (
            "criteria — key answer-evaluation dimensions the answerer weighs "
            "(e.g., \"average starting salary\", \"scholarship availability\")"
        )
        atom_word = "criterion"
        atoms_word = "criteria"
        node_type = "concept"
        cf_verb = "weigh"
        p4_intro = (
            "P4) For each other answer A_i, find the set R_i of criterion seeds to REMOVE from the full atom set "
            "such that, evaluating only the remaining criteria, A_i beats A*. Then S(A_i) is the full set minus R_i.\n"
            "          - Each c in R_i must be a criterion that specifically favors A* over A_i, or specifically hurts A_i. "
            "For each removal, briefly say why in one clause.\n"
            "          - Do NOT remove criteria just because they seem irrelevant to A_i — only remove ones that actively work against A_i."
        )
        edge_semantics = (
            "Edge semantics: c->a means \"c is in S(a), the criterion subset under which a is the top pick.\" "
            "A* is wired to ALL c's (S(A*) = full seed set). Each alternative A_i is wired to ALL c's EXCEPT its "
            "removal set R_i, so most c's connect to most answers. The DIFFERENTIATION signal lives in the MISSING "
            "edges — c ↛ A_i means c is in R_i, a criterion that specifically favors A* over A_i."
        )
    else:
        layer3_desc = (
            "source directives — filters on WHICH sources to prioritize when "
            "answering the sub-question (temporal, geographic, trustworthiness, "
            "or perspective)"
        )
        atom_word = "directive"
        atoms_word = "directives"
        node_type = "directive"
        cf_verb = "apply"
        p4_intro = (
            "P4) For each other answer A_i, find the set R_i of directive seeds to DROP from the full directive set "
            "such that, with the source pool broadened by dropping those filters (letting in sources that were "
            "previously excluded), the evidence now supports A_i over A*. Then S(A_i) is the full set minus R_i.\n"
            "          - Each d in R_i must be a directive whose filtering specifically excludes sources that would "
            "have supported A_i, or whose enforcement specifically keeps in sources that favor A*. For each drop, "
            "briefly say which sources come in / go out and why that flips the winner.\n"
            "          - Do NOT drop a directive just because it seems tangential — only drop ones whose filtering "
            "materially shapes the source pool against A_i."
        )
        edge_semantics = (
            "Edge semantics: d->a means \"d is in S(a), the directive subset under which a is the top pick given "
            "the sources those directives admit.\" A* is wired to ALL d's (S(A*) = full directive set — A* wins "
            "when every filter is enforced). Each alternative A_i is wired to ALL d's EXCEPT its removal set R_i, "
            "where R_i is the directives that must be RELAXED (source pool broadened) for A_i to win. The "
            "DIFFERENTIATION signal lives in the MISSING edges — d ↛ A_i means d is in R_i, a filter whose "
            "enforcement specifically props up A* over A_i."
        )

    n_seeds = len(seeded_atoms) if seeded_atoms else 0
    n_ans = len(seeded_answers) if seeded_answers else 0

    header = textwrap.dedent(f"""\
    Build a 4-layer reasoning graph for the QUESTION using the provided SOURCES to match your step by step reasoning process.
    The graph should be organized with the following layer structure.

    Layers:
    - Layer 1: the question (id "q").
    - Layer 2: distinct sub-questions (ids sq1..).
    - Layer 3: {layer3_desc} (ids c1..).
    - Layer 4: the fixed plausible answers listed at the end (ids a1..).

    First think step by step in prose, then emit the JSON in a ```json fenced block at the END of your response. Do not skip the reasoning.

    THINK OUT LOUD (write these before the JSON):
      P1) Restate the question in one line; note what distinguishes the plausible answers from each other.
      P2) Group the Layer 3 seeds into 2-4 sub-questions.
      P3) S(A*) = ALL Layer 3 seeds. A* is the top-1 pick when the answerer {cf_verb}s the full seed set, so by definition every seed is in S(A*) — every Layer 3 seed gets a c->A* edge. No removal for A*.
      {p4_intro}
          - SIZE R_i TO WHATEVER IT TAKES. If a subtle counterfactual flips A_i with 1-2 removals, use that. If A* is heavily favored and the flip requires stripping many {atoms_word}, use a larger R_i. Do NOT settle for a small R_i that fails to flip.
          - A working larger R_i is more useful than a broken smaller one. Prefer smaller when it truly flips; grow the set as needed.
          - No two R_i may be identical; every R_i must be non-empty.
      P5) Sanity-check: each Layer 3 node has >=1 sq parent AND >=1 answer child; no two answers share a supporter set.

    Then emit:
    ```json
    { "{" } "nodes": [{ "{" } "id": "...", "layer": "1|2|3|4", "type": "question|subq|{node_type}|answer", "label": "...", "description": "..." { "}" }, ...],
      "edges": [{ "{" } "source": "...", "target": "..." { "}" }, ...] { "}" }
    ```

    {edge_semantics}

    Rules:
    1) Edges only between adjacent layers: q->sq, sq->c, c->a. No skips.
    2) NO ORPHANS: every Layer 3 node MUST have >=1 incoming sq edge.
    3) Answer labels are short — no filler ("prevention", not "the best treatment is prevention").
    4) No two answers may share the same supporter set. Every S(A_i) besides S(A*) must be a proper subset of the full {atom_word} set (R_i non-empty). Size R_i to what the flip actually requires — no fixed budget.
    """).replace('{ "{" }', '{').replace('{ "}" }', '}')

    # Seeds + final reminder together at the tail (recency bias).
    seed_tail = ""
    if seeded_atoms:
        seed_list = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(seeded_atoms))
        if mode == "criteria":
            out_hint = "to the a(s) it GENUINELY FAVORS (out)"
        else:
            out_hint = (
                "to the a(s) whose supporting sources SURVIVE this filter (out) — "
                "i.e., every a except those that need this directive relaxed to win"
            )
        seed_tail += (
            f"\nLAYER 3 (exactly these {n_seeds}, verbatim labels, no drops, no rename, "
            f"no additions):\n{seed_list}\n"
            f"Each MUST be wired to one sq (in) and {out_hint}.\n"
        )
    if seeded_answers:
        ans_list = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(seeded_answers))
        seed_tail += (
            f"\nLAYER 4 (exactly these {n_ans} answers, verbatim labels, nothing else):\n"
            f"{ans_list}\n"
        )
    if baseline_answer:
        seed_tail += (
            f"\nBASELINE ANSWER A* (top-ranked when the answerer considers ALL Layer 3 "
            f"seeds together):\n  {baseline_answer}\n"
        )

    body = f"\nSOURCES:\n{sources_text}\n\nQUESTION:\n{question}\n"

    return header + body + seed_tail + "\nOUTPUT:\n"


def select_atoms(row: Dict[str, Any], mode: str) -> List[str]:
    if mode == "criteria":
        return list(row.get("criteria") or [])
    if mode == "pragmas":
        return flatten_pragmas(row.get("pragmas") or [])
    raise ValueError(f"unknown mode: {mode!r}")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_json_comments(text: str) -> str:
    """Remove // line comments and /* */ block comments that models sometimes
    inject into JSON output. Strict JSON forbids both."""
    text = _BLOCK_COMMENT_RE.sub("", text)
    return _LINE_COMMENT_RE.sub("", text)


def parse_graph_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the LLM's graph output. Returns None on failure.

    Accepts either (a) raw JSON, (b) JSON wrapped in a single ```json fence
    (the CoT-friendly format), or (c) prose reasoning followed by a fenced
    JSON block at the end. Prefers the LAST fenced block so reasoning that
    mentions {...} in prose doesn't confuse the parser. Strips //-style and
    /*..*/ comments before parsing since models like to annotate the JSON.
    """
    text = (raw or "").strip()

    # Prefer the last fenced ```json { ... } ``` block — this handles
    # step-by-step reasoning followed by the graph JSON.
    matches = list(_JSON_FENCE_RE.finditer(text))
    candidates: List[str] = [m.group(1) for m in reversed(matches)]

    # Fallback: whole-response strip-fence + last {...} slice.
    stripped = _strip_code_fences(text)
    candidates.append(stripped)
    end = stripped.rfind("}")
    start = stripped.rfind("{", 0, end) if end != -1 else -1
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for cand in candidates:
        for text_try in (cand, _strip_json_comments(cand)):
            try:
                parsed = json.loads(text_try)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "nodes" in parsed and "edges" in parsed:
                return parsed
    return None


def generate_row_graph(
    *,
    client: Any,
    provider: str,
    model: str,
    row: Dict[str, Any],
    mode: str,
    out_dir: Path,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> None:
    atoms = select_atoms(row, mode)
    sources_text = stringify_sources(row.get("sources"))
    question = row.get("query") or ""
    plausible = list(row.get("plausible_answers") or [])

    baseline_idx, baseline_label = pick_answer_with_all_atoms(
        client=client, provider=provider, model=model,
        question=question, sources=sources_text,
        atoms=atoms, plausible_answers=plausible, mode=mode,
        temperature=0.0, max_tokens=256, max_retries=max_retries,
    )

    prompt = build_graph_of_thoughts_prompt(
        sources_text=sources_text,
        question=question,
        mode=mode,
        seeded_atoms=atoms,
        seeded_answers=plausible,
        baseline_answer=baseline_label,
    )
    messages = [{"role": "user", "content": prompt}]

    graph = None
    raw = ""
    attempts = 0
    for attempts in range(1, max_retries + 1):
        raw = _invoke_chat_completion(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        graph = parse_graph_json(raw)
        if graph is not None:
            break
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "Your previous answer did not contain a parseable JSON object "
                "with 'nodes' and 'edges'. Keep any reasoning brief, then emit the "
                "graph inside a single ```json ...``` fenced block at the end."
            )},
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_id": row.get("id"),
        "query": question,
        "mode": mode,
        "atoms": atoms,
        "plausible_answers": plausible,
        "baseline_answer": baseline_label,
        "baseline_index": baseline_idx,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "attempts": attempts,
        "graph": graph,
    }
    if graph is None:
        payload["raw"] = raw
        payload["parse_error"] = True

    # A re-run of a failed row would overwrite the raw output that explains the
    # failure, so keep the old payload before replacing it. Without this you
    # cannot tell afterwards whether the model or the parser was at fault.
    out_path = out_dir / "graph.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        if prev.get("parse_error"):
            out_path.replace(out_dir / "graph.failed.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def graph_complete(out_dir: Path) -> bool:
    p = out_dir / "graph.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        return d.get("graph") is not None
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", type=Path, required=True,
                    help="Path to msmarco-data.json or sonnet-dataset.json")
    ap.add_argument("--mode", choices=("criteria", "pragmas"), required=True)
    ap.add_argument("--run-name", required=True,
                    help="Subdir under graphs/ for this run's outputs")
    ap.add_argument("--provider", choices=("openai", "vllm", "local"), default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--openai-base-url", default=None,
                    help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 for a "
                         "vLLM server. Also read from OPENAI_BASE_URL env var. "
                         "Only used when provider=openai.")
    ap.add_argument("--api-key", default=None,
                    help="Override OPENAI_API_KEY. Local servers usually ignore it.")
    ap.add_argument("--qids", type=parse_qid_filter, default=None,
                    help="Comma-separated qids to include (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N rows after filtering")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="GPUs for vLLM (vllm provider only)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="Cap tokens per request (prompt + output) for vLLM. vLLM reserves "
                         "KV cache for one request at the model's full context, which large "
                         "windows make unaffordable on smaller cards; 8192 covers these prompts.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="Fraction of GPU memory vLLM may reserve (vllm provider only)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run rows whose graph.json already parsed successfully")
    args = ap.parse_args()

    rows = load_rows(args.source)
    if args.qids is not None:
        rows = [r for r in rows if r["id"] in args.qids]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No rows to process.")
        return

    output_root = GRAPHS_ROOT / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    client = create_llm_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.openai_base_url,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    skipped = completed = failed = 0
    for i, row in enumerate(rows, 1):
        qid = row["id"]
        out_dir = output_root / str(qid)
        if not args.force and graph_complete(out_dir):
            print(f"[{i}/{len(rows)}] qid={qid} skip (graph exists)")
            skipped += 1
            continue

        atoms = select_atoms(row, args.mode)
        if not atoms:
            print(f"[{i}/{len(rows)}] qid={qid} skip (no {args.mode})")
            skipped += 1
            continue

        print(f"[{i}/{len(rows)}] qid={qid} — {len(atoms)} {args.mode} atoms")
        try:
            generate_row_graph(
                client=client,
                provider=args.provider,
                model=args.model,
                row=row,
                mode=args.mode,
                out_dir=out_dir,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
            if graph_complete(out_dir):
                completed += 1
            else:
                print(f"  parse failed after {args.max_retries} attempts (raw saved)")
                failed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\nDone. completed={completed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
