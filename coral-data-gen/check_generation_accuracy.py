"""
Generation-accuracy check for pre-generated Graphs of Thoughts.

For each graph in graphs/<run-name>/*/graph.json, and for each Layer 4 answer A_i
in that graph:
  1) Take S(A_i) = the Layer 3 criteria/pragmas connected to A_i via c* -> a*.
  2) Rank the row's plausible_answers using ONLY those criteria/pragmas, via the
     exact create_full_ranking_prompt (or _pragmas) from generate_dataset.py.
  3) Score 1 if the top-1 ranked answer equals A_i, else 0.

Per-graph accuracy = matches / len(plausible_answers).
Cumulative accuracy = sum(matches) / sum(plausible_answers).

The graph's `mode` field determines whether to use the criteria or pragmas
ranking prompt. Sources / query / plausible_answers are read from the base
dataset row.

Usage:
    python check_generation_accuracy.py \\
        --source datasets/msmarco-data.json \\
        --run-dir graphs/llama31-8b-msmarco-criteria \\
        --provider vllm --model meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel-size 1

    # OpenAI (much cheaper for iteration):
    python check_generation_accuracy.py \\
        --source datasets/msmarco-data.json --run-dir graphs/... \\
        --provider openai --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generate_dataset import create_llm_client
from graph_ops import Graph, atoms_for_answer, load_graph, pick_answer_with_all_atoms
from run_lattices import load_rows, stringify_sources


# A picker takes (supporters, plausible_answers) and returns (top1_index, top1_label).
Picker = Any  # Callable[[List[str], List[str]], Tuple[int, str]]


def score_graph(
    *,
    picker: Picker,
    graph: Graph,
    plausible_answers: List[str],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Return (matches, total, per_answer_records).

    picker(supporters, plausible_answers) -> (top1_index, top1_label). Swap
    between an LLM-backed picker and a lattice-lookup picker without touching
    this function.
    """
    answers = graph.answers()
    matches = 0
    records: List[Dict[str, Any]] = []

    for a in answers:
        answer_label = a.get("label", "")
        supporters = sorted(atoms_for_answer(graph, a["id"]))

        # Sanity: the graph's answer label must be one of the plausible answers
        # (that's how we generated it). If it isn't, we can still test it, but
        # top-1 will never match, so score it as an ERROR row.
        if answer_label not in plausible_answers:
            records.append({
                "answer_id": a["id"], "answer_label": answer_label,
                "supporters": supporters, "picked": None,
                "matched": False, "error": "answer_not_in_plausible",
            })
            continue

        try:
            picked_idx, picked_label = picker(supporters, plausible_answers)
        except Exception as exc:
            records.append({
                "answer_id": a["id"], "answer_label": answer_label,
                "supporters": supporters, "picked": None,
                "matched": False, "error": f"llm_or_parse: {exc}",
            })
            continue

        matched = picked_label == answer_label
        if matched:
            matches += 1
        records.append({
            "answer_id": a["id"], "answer_label": answer_label,
            "supporters": supporters,
            "picked_index": picked_idx, "picked_label": picked_label,
            "matched": matched, "error": None,
        })

    return matches, len(answers), records


def _fmt_pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "N/A"


# ------------------------------------------------------------------ #
# Pickers                                                            #
# ------------------------------------------------------------------ #

def build_llm_picker(
    *, client: Any, provider: str, model: str, mode: str,
    question: str, sources: str,
    temperature: float, max_tokens: int, max_retries: int,
) -> Picker:
    """Closure that binds row-level context so score_graph's picker signature
    stays supporters+plausible."""
    def _pick(supporters: List[str], plausible_answers: List[str]) -> Tuple[int, str]:
        return pick_answer_with_all_atoms(
            client=client, provider=provider, model=model,
            question=question, sources=sources,
            atoms=supporters, plausible_answers=plausible_answers,
            mode=mode, temperature=temperature, max_tokens=max_tokens,
            max_retries=max_retries,
        )
    return _pick


def build_lattice_picker(lattice_row: Dict[str, Any]) -> Picker:
    """Zero-LLM picker: look up the pre-computed ranking for the supporter set.
    Lattice combo_key is a comma-joined sorted list of active atoms (empty
    string for empty set); value is the ranked plausible_answer labels.
    """
    answers_by_combo: Dict[str, List[str]] = lattice_row.get("raw_answers") or lattice_row.get("answers", {})
    lattice_atoms = set(lattice_row.get("criteria") or lattice_row.get("pragmas") or [])

    def _pick(supporters: List[str], plausible_answers: List[str]) -> Tuple[int, str]:
        # Drop any supporter that isn't in the lattice's atom space (would
        # happen if the graph generated an extra c-node beyond the seeds).
        active = sorted(a for a in supporters if a in lattice_atoms)
        key = ", ".join(active)
        ranking = answers_by_combo.get(key) or answers_by_combo.get(",".join(active))
        if ranking is None:
            raise KeyError(f"combo not in lattice: {key!r}")
        top = ranking[0]
        try:
            idx = plausible_answers.index(top)
        except ValueError:
            idx = -1
        return idx, top
    return _pick


def load_lattice_file(path: Path) -> Dict[str, Any]:
    """Return {qid_str: row_dict} — skips the "_meta" entry if present."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if k != "_meta"}


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", type=Path, required=True,
                    help="Base dataset (msmarco-data.json or sonnet-dataset.json)")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="graphs/<run-name> — iterates all */graph.json under it")
    ap.add_argument("--lattice-file", type=Path, default=None,
                    help="Pre-computed lattice (e.g. datasets/qwen3-32b/msmarco-criteria-qwen3-32.json). "
                         "If provided, skip the LLM entirely and look up rankings from this file. "
                         "Free and fast — recommended when a matching lattice exists.")
    ap.add_argument("--provider", choices=("openai", "vllm", "local"), default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--qids", type=lambda s: {int(x) for x in s.split(",")},
                    default=None, help="Only score these qids (comma-separated)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N graphs after filtering")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write full per-graph JSON report here (default: <run-dir>/generation-accuracy.json)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-answer detail; only print per-graph summary + cumulative")
    args = ap.parse_args()

    # Index dataset by qid
    rows_by_qid = {r["id"]: r for r in load_rows(args.source)}

    # Discover graphs
    paths = sorted(args.run_dir.glob("*/graph.json"))
    if not paths:
        raise SystemExit(f"no graph.json under {args.run_dir}")

    # Filter
    def qid_of(p: Path) -> int:
        try:
            return int(p.parent.name)
        except ValueError:
            return -1
    paths = [p for p in paths if qid_of(p) >= 0]
    if args.qids is not None:
        paths = [p for p in paths if qid_of(p) in args.qids]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no graphs left after filtering")

    lattice_rows: Optional[Dict[str, Any]] = None
    client = None
    if args.lattice_file is not None:
        lattice_rows = load_lattice_file(args.lattice_file)
        print(f"Using lattice file (no LLM calls): {args.lattice_file} "
              f"[{len(lattice_rows)} qids indexed]")
    else:
        client = create_llm_client(
            provider=args.provider,
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    total_matches = 0
    total_answers = 0
    skipped = 0
    graph_reports: List[Dict[str, Any]] = []

    for i, p in enumerate(paths, 1):
        qid = qid_of(p)
        row = rows_by_qid.get(qid)
        if row is None:
            print(f"[{i}/{len(paths)}] qid={qid} SKIP: not in {args.source}")
            skipped += 1
            continue
        try:
            graph = load_graph(p)
        except ValueError as exc:
            print(f"[{i}/{len(paths)}] qid={qid} SKIP: {exc}")
            skipped += 1
            continue

        mode = graph.meta.get("mode") or "criteria"
        plausible = row["plausible_answers"]
        sources = stringify_sources(row["sources"])

        if lattice_rows is not None:
            lat_row = lattice_rows.get(str(qid))
            if lat_row is None:
                print(f"[{i}/{len(paths)}] qid={qid} SKIP: no lattice entry")
                skipped += 1
                continue
            if lat_row.get("halted"):
                print(f"[{i}/{len(paths)}] qid={qid} SKIP: lattice halted")
                skipped += 1
                continue
            picker = build_lattice_picker(lat_row)
        else:
            picker = build_llm_picker(
                client=client, provider=args.provider, model=args.model, mode=mode,
                question=row["query"], sources=sources,
                temperature=args.temperature, max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )

        matches, total, records = score_graph(
            picker=picker, graph=graph, plausible_answers=plausible,
        )
        total_matches += matches
        total_answers += total

        pct = _fmt_pct(matches, total)
        print(f"[{i}/{len(paths)}] qid={qid} mode={mode}  {matches}/{total} supported ({pct})")
        if not args.quiet:
            for r in records:
                mark = "✓" if r["matched"] else "✗"
                extra = f" ERROR={r['error']}" if r.get("error") else ""
                picked = r.get("picked_label") or "-"
                print(f"    {mark} {r['answer_id']}  ({len(r['supporters'])} supporters)  -> picked: {picked!r}{extra}")

        graph_reports.append({
            "qid": qid, "mode": mode, "matches": matches, "total": total,
            "records": records,
        })

    print()
    print("=" * 60)
    print(f"CUMULATIVE  ({len(graph_reports)} graphs, {skipped} skipped)")
    print(f"  {total_matches}/{total_answers} answers supported "
          f"({_fmt_pct(total_matches, total_answers)})")
    # Also: per-graph accuracy = fraction of graphs where ALL answers matched
    fully = sum(1 for g in graph_reports if g["matches"] == g["total"] and g["total"] > 0)
    print(f"  {fully}/{len(graph_reports)} graphs with 100% support "
          f"({_fmt_pct(fully, len(graph_reports))})")

    out = args.output or (args.run_dir / "generation-accuracy.json")
    out.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "source": str(args.source),
        "provider": args.provider,
        "model": args.model,
        "cumulative": {
            "graphs": len(graph_reports),
            "skipped": skipped,
            "matches": total_matches,
            "total": total_answers,
            "answer_accuracy": (total_matches / total_answers) if total_answers else None,
            "graphs_100pct": fully,
            "graph_full_accuracy": (fully / len(graph_reports)) if graph_reports else None,
        },
        "per_graph": graph_reports,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote report → {out}")


if __name__ == "__main__":
    main()
