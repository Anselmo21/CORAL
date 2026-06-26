"""
Run lattice generation across every row of a base dataset.

Loads either msmarco-data.json (dict with "rows") or sonnet-dataset.json
(top-level list), normalizes the field names, then calls
generate_dataset.generate_dataset(...) once per row.

Each row's lattice lands at lattices/<run_name>/<qid>/{config.json, answers.json}.
Resume-safe: a row is skipped if its config.json already reports combos_run ==
combos_total. The underlying generator also resumes per-combo when invoked with
fill_missing=True.

Usage:
    python run_lattices.py --source datasets/msmarco-data.json \
        --mode pragmas --run-name msmarco-pragmas-openai \
        --provider openai --model gpt-4o-mini

    python run_lattices.py --source datasets/sonnet-dataset.json \
        --mode criteria --run-name sonnet-criteria-vllm \
        --provider vllm --model Qwen/Qwen3-8B

For vLLM, you might need to specify which GPU to use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from generate_dataset import create_llm_client, generate_dataset


LATTICES_ROOT = Path("lattices")


def load_rows(source: Path) -> List[Dict[str, Any]]:
    """Return a normalized list of rows from either msmarco-data.json or
    sonnet-dataset.json. Each row has keys: id, query, sources, criteria, pragmas,
    plausible_answers."""
    raw = json.loads(source.read_text(encoding="utf-8"))
    items = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
    rows = []
    for r in items:
        rows.append({
            "id": r.get("query_id", r.get("id")),
            "query": r.get("query", r.get("user_query")),
            "sources": r.get("sources", r.get("retrieved_sources")),
            "criteria": r.get("criteria") or [],
            "pragmas": r.get("pragmas") or [],
            "plausible_answers": r.get("plausible_answers") or [],
        })
    return rows


def flatten_pragmas(pragmas: Any) -> List[str]:
    """Pragmas come in two shapes:
      - msmarco: dict {temporal: [...], geographic: [...], ...}
      - sonnet:  flat list of "P{N} (CATEGORY): value" strings
    Normalize to a flat list of strings.
    """
    if isinstance(pragmas, dict):
        return [f"{cat}: {v}" for cat, values in pragmas.items() for v in (values or [])]
    if isinstance(pragmas, list):
        return [str(p) for p in pragmas]
    return []


def stringify_sources(sources: Any) -> str:
    """msmarco sources are already a string; sonnet retrieved_sources is a list."""
    if isinstance(sources, str):
        return sources
    if isinstance(sources, list):
        return "\n\n".join(str(s) for s in sources)
    return str(sources or "")


def lattice_complete(out_dir: Path) -> bool:
    cfg = out_dir / "config.json"
    if not cfg.exists():
        return False
    try:
        c = json.loads(cfg.read_text())
        return c.get("combos_run") == c.get("combos_total") and not c.get("halted")
    except Exception:
        return False


def select_atoms(row: Dict[str, Any], mode: str) -> List[str]:
    if mode == "criteria":
        return list(row["criteria"])
    if mode == "pragmas":
        return flatten_pragmas(row["pragmas"])
    raise ValueError(f"unknown mode: {mode!r}")


def parse_qid_filter(spec: str) -> set:
    return {int(x) for x in spec.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="Path to msmarco-data.json or sonnet-dataset.json")
    ap.add_argument("--mode", choices=("criteria", "pragmas"), required=True)
    ap.add_argument("--run-name", required=True,
                    help="Subdir under lattices/ for this run's outputs")
    ap.add_argument("--provider", choices=("openai", "vllm", "local"), default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--qids", type=parse_qid_filter, default=None,
                    help="Comma-separated qids to include (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N rows after filtering")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--fill-missing", action="store_true",
                    help="Resume per-combo within each row (not just per-row)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a row's lattice is already complete")
    args = ap.parse_args()

    rows = load_rows(args.source)
    if args.qids is not None:
        rows = [r for r in rows if r["id"] in args.qids]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No rows to process.")
        return

    output_root = LATTICES_ROOT / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    # Build the LLM client once and reuse — critical for local/vllm.
    client = create_llm_client(provider=args.provider, model=args.model)

    skipped = completed = failed = 0
    for i, row in enumerate(rows, 1):
        qid = row["id"]
        out_dir = output_root / str(qid)
        if not args.force and lattice_complete(out_dir):
            print(f"[{i}/{len(rows)}] qid={qid} skip (complete)")
            skipped += 1
            continue

        atoms = select_atoms(row, args.mode)
        if not atoms:
            print(f"[{i}/{len(rows)}] qid={qid} skip (no {args.mode})")
            skipped += 1
            continue

        plausible = row["plausible_answers"]
        if not plausible:
            print(f"[{i}/{len(rows)}] qid={qid} skip (no plausible_answers)")
            skipped += 1
            continue

        print(f"\n[{i}/{len(rows)}] qid={qid} — {len(atoms)} atoms, "
              f"{2**len(atoms)} combos")
        try:
            generate_dataset(
                dataset_name=f"{args.run_name}/{qid}",
                question=row["query"],
                criteria=atoms,
                plausible_answers=plausible,
                sources=stringify_sources(row["sources"]),
                output_dir=str(LATTICES_ROOT),
                model=args.model,
                mode=args.mode,
                provider=args.provider,
                client=client,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retries=args.max_retries,
                fill_missing=args.fill_missing,
                batch_size=args.batch_size,
            )
            completed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\nDone. completed={completed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
