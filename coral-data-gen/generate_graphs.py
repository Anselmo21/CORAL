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
    python generate_graphs.py --source datasets/msmarco-data.json \\
        --mode criteria --run-name msmarco-criteria-gpt4o-graphs \\
        --provider openai --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from generate_dataset import (
    _invoke_chat_completion,
    _strip_code_fences,
    create_llm_client,
)
from run_lattices import flatten_pragmas, load_rows, parse_qid_filter, stringify_sources


GRAPHS_ROOT = Path("graphs")


def build_graph_of_thoughts_prompt(
    sources_text: str,
    question: str,
    mode: str = "criteria",
    seeded_atoms: Optional[List[str]] = None,
    seeded_answers: Optional[List[str]] = None,
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
    else:
        layer3_desc = (
            "source directives — filters on WHICH sources to prioritize when "
            "answering the sub-question (temporal, geographic, trustworthiness, "
            "or perspective); NOT answer-evaluation criteria, they constrain "
            "the evidence the answer is drawn from"
        )

    n_seeds = len(seeded_atoms) if seeded_atoms else 0
    n_ans = len(seeded_answers) if seeded_answers else 0
    seed_max = n_seeds + 2  # allow up to 2 extras

    header = textwrap.dedent(f"""\
    Build a 4-layer reasoning graph for the QUESTION using the CONTEXT.

    Layers:
    - Layer 1: the question (id "q").
    - Layer 2: 3-4 distinct sub-questions (ids sq1..).
    - Layer 3: {layer3_desc} (ids c1..).
    - Layer 4: the fixed plausible answers listed at the end (ids a1..).

    Output ONLY this JSON, no prose, no code fences. Edges have no relation label — a c->a edge means the criterion supports the answer; interpret every edge as "the parent supports/informs the child".
    { "{" } "nodes": [{ "{" } "id": "...", "layer": "1|2|3|4", "type": "question|subq|criterion|answer", "label": "...", "description": "..." { "}" }, ...],
      "edges": [{ "{" } "source": "...", "target": "..." { "}" }, ...] { "}" }

    Rules:
    1) Edges only between adjacent layers: q->sq, sq->c, c->a. No skips.
    2) Every sq covers >=2 c nodes; every a is reached by >=2 c nodes.
    3) NO ORPHANS: every c node MUST have >=1 incoming sq edge AND >=1 outgoing a edge. Before emitting, walk c1, c2, ... and confirm both edges exist for each.
    4) Answer labels are short — no filler ("prevention", not "the best treatment is prevention").
    """).replace('{ "{" }', '{').replace('{ "}" }', '}')

    # Seeds + final reminder together at the tail (recency bias).
    seed_tail = ""
    if seeded_atoms:
        seed_list = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(seeded_atoms))
        seed_tail += (
            f"\nLAYER 3 SEEDS ({n_seeds} required, verbatim labels, no drops, no rename; "
            f"up to {seed_max} total allowed):\n{seed_list}\n"
            f"Each seed MUST be wired to one sq (in) and >=1 a (out).\n"
        )
    if seeded_answers:
        ans_list = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(seeded_answers))
        seed_tail += (
            f"\nLAYER 4 (exactly these {n_ans} answers, verbatim labels, nothing else):\n"
            f"{ans_list}\n"
        )

    body = f"\nCONTEXT:\n{sources_text}\n\nQUESTION:\n{question}\n"

    return header + body + seed_tail + "\nOUTPUT:\n"


def select_atoms(row: Dict[str, Any], mode: str) -> List[str]:
    if mode == "criteria":
        return list(row.get("criteria") or [])
    if mode == "pragmas":
        return flatten_pragmas(row.get("pragmas") or [])
    raise ValueError(f"unknown mode: {mode!r}")


def parse_graph_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the LLM's graph output. Returns None on failure."""
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to isolate the first top-level {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
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

    prompt = build_graph_of_thoughts_prompt(
        sources_text=sources_text,
        question=question,
        mode=mode,
        seeded_atoms=atoms,
        seeded_answers=plausible,
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
                "Your previous answer was not a valid JSON object with 'nodes' and "
                "'edges'. Return ONLY the JSON object, no code fences, no prose."
            )},
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_id": row.get("id"),
        "query": question,
        "mode": mode,
        "atoms": atoms,
        "plausible_answers": plausible,
        "model": model,
        "provider": provider,
        "temperature": temperature,
        "attempts": attempts,
        "graph": graph,
    }
    if graph is None:
        payload["raw"] = raw
        payload["parse_error"] = True
    (out_dir / "graph.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    ap.add_argument("--qids", type=parse_qid_filter, default=None,
                    help="Comma-separated qids to include (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N rows after filtering")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="GPUs for vLLM (vllm provider only)")
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
        tensor_parallel_size=args.tensor_parallel_size,
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
