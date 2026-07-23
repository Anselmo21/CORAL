"""
Graph-of-Thoughts analysis pipeline for a single (row, mode) pair.

Steps:
  1) Prompt the LLM to pick the best answer from the row's plausible_answers,
     given ALL the row's atoms (criteria or pragmas). This is the "original
     answer" A*. Uses the existing full-ranking prompts from generate_dataset.py
     and takes the top-1.
  2) Load the pre-generated GoT for this (row, mode) from graphs/<run>/<qid>/graph.json.
     (Assumes Layer 4 was seeded with plausible_answers so A* is a node.)
  3) Read S(A*) = the Layer 3 labels connected to A* via c* -> a* edges.
  4) For each other Layer 4 node A_i, compute S(A_i) and pick the alternative
     that minimizes |S(A*) Δ S(A_i)| = removed + added. Ties broken by (fewer
     removals, fewer additions, then lexicographic id).

Also provides plain-text rendering: pretty_print(graph).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from generate_dataset import (
    _invoke_chat_completion,
    _repair_json_array,
    _strip_code_fences,
    create_full_ranking_prompt,
    create_full_ranking_prompt_pragmas,
)


# ------------------------------------------------------------------ #
# Loading                                                            #
# ------------------------------------------------------------------ #

@dataclass
class Graph:
    nodes_by_id: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, str]]
    meta: Dict[str, Any]

    def by_layer(self, layer: str) -> List[Dict[str, Any]]:
        return [n for n in self.nodes_by_id.values() if str(n.get("layer")) == str(layer)]

    def answers(self) -> List[Dict[str, Any]]:
        return self.by_layer("4")

    def layer3(self) -> List[Dict[str, Any]]:
        return self.by_layer("3")

    def label(self, node_id: str) -> str:
        node = self.nodes_by_id.get(node_id)
        return (node or {}).get("label", node_id)


def load_graph(path: Path) -> Graph:
    """Load a graph.json produced by generate_graphs.py."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    g = data.get("graph")
    if not g:
        raise ValueError(f"{path}: no graph (parse_error or missing)")
    nodes_by_id = {n["id"]: n for n in g["nodes"]}
    return Graph(nodes_by_id=nodes_by_id, edges=list(g["edges"]), meta=data)


# ------------------------------------------------------------------ #
# Step 1: pick the original answer using ALL atoms                   #
# ------------------------------------------------------------------ #

def pick_answer_with_all_atoms(
    *,
    client: Any,
    provider: str,
    model: str,
    question: str,
    sources: str,
    atoms: List[str],
    plausible_answers: List[str],
    mode: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> Tuple[int, str]:
    """Rank plausible_answers with all atoms active; return (top1_index, top1_text)."""
    if mode == "criteria":
        prompt = create_full_ranking_prompt(question, atoms, sources, plausible_answers)
    elif mode == "pragmas":
        prompt = create_full_ranking_prompt_pragmas(question, atoms, sources, plausible_answers)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    n = len(plausible_answers)
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_retries):
        raw = _invoke_chat_completion(
            client=client, provider=provider, model=model,
            messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        try:
            ranking = json.loads(_repair_json_array(_strip_code_fences(raw)))
        except Exception:
            ranking = None
        if (
            isinstance(ranking, list)
            and len(ranking) == n
            and set(ranking) == set(range(n))
        ):
            top1 = int(ranking[0])
            return top1, plausible_answers[top1]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"Your previous answer was invalid. Return ONLY a JSON array of "
                f"exactly {n} integers, a permutation of 0..{n-1}, best first."
            )},
        ]
    raise RuntimeError(f"could not parse ranking after {max_retries} retries")


# ------------------------------------------------------------------ #
# Steps 3-4: graph traversal                                         #
# ------------------------------------------------------------------ #

def find_answer_node(graph: Graph, answer_text: str) -> Optional[str]:
    """Return the Layer 4 node id whose label matches answer_text exactly, else None."""
    for a in graph.answers():
        if a.get("label") == answer_text:
            return a["id"]
    return None


def atoms_for_answer(graph: Graph, answer_id: str) -> Set[str]:
    """Set of Layer 3 labels connected to answer_id via c* -> a* edges."""
    l3_ids = {n["id"] for n in graph.layer3()}
    supporting_ids = {
        e["source"] for e in graph.edges
        if e.get("target") == answer_id and e.get("source") in l3_ids
    }
    return {graph.label(i) for i in supporting_ids}


@dataclass
class AlternativeMatch:
    alt_id: str
    alt_label: str
    added: List[str]     # in S(alt) but not in S(orig)
    removed: List[str]   # in S(orig) but not in S(alt)
    distance: int        # len(added) + len(removed)


def closest_alternative(graph: Graph, original_id: str) -> Optional[AlternativeMatch]:
    """Pick the Layer 4 node (other than original_id) minimizing symmetric-difference
    of connected Layer 3 sets. Ties: fewer removals, then fewer additions, then id."""
    s_orig = atoms_for_answer(graph, original_id)
    best: Optional[AlternativeMatch] = None
    for a in graph.answers():
        if a["id"] == original_id:
            continue
        s_alt = atoms_for_answer(graph, a["id"])
        added = sorted(s_alt - s_orig)
        removed = sorted(s_orig - s_alt)
        cand = AlternativeMatch(
            alt_id=a["id"], alt_label=a.get("label", a["id"]),
            added=added, removed=removed, distance=len(added) + len(removed),
        )
        key = (cand.distance, len(cand.removed), len(cand.added), cand.alt_id)
        best_key = (best.distance, len(best.removed), len(best.added), best.alt_id) if best else None
        if best is None or key < best_key:  # type: ignore[operator]
            best = cand
    return best


# ------------------------------------------------------------------ #
# Full pipeline                                                      #
# ------------------------------------------------------------------ #

@dataclass
class PipelineResult:
    original_index: int
    original_label: str
    original_id: str
    original_atoms: List[str]
    alternative: Optional[AlternativeMatch]


def run_pipeline(
    *,
    client: Any,
    provider: str,
    model: str,
    graph_path: Path,
    question: str,
    sources: str,
    atoms: List[str],
    plausible_answers: List[str],
    mode: str,
    temperature: float = 0.0,
) -> PipelineResult:
    """Steps 1-4 end-to-end for one (row, mode)."""
    graph = load_graph(graph_path)

    top_idx, top_label = pick_answer_with_all_atoms(
        client=client, provider=provider, model=model,
        question=question, sources=sources, atoms=atoms,
        plausible_answers=plausible_answers, mode=mode,
        temperature=temperature,
    )
    node_id = find_answer_node(graph, top_label)
    if node_id is None:
        raise ValueError(
            f"Original answer {top_label!r} does not appear as a Layer 4 node in "
            f"{graph_path}. Was the graph generated with seeded_answers=plausible_answers?"
        )

    s_orig = sorted(atoms_for_answer(graph, node_id))
    alt = closest_alternative(graph, node_id)
    return PipelineResult(
        original_index=top_idx,
        original_label=top_label,
        original_id=node_id,
        original_atoms=s_orig,
        alternative=alt,
    )


# ------------------------------------------------------------------ #
# Validation                                                         #
# ------------------------------------------------------------------ #

@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def format(self) -> str:
        head = "OK" if self.ok else f"FAIL ({len(self.errors)} errors)"
        lines = [head]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  warn:  {w}")
        return "\n".join(lines)


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, drop trailing punctuation for fuzzy label match."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.\s]+$", "", s)
    return s


def validate_graph(
    graph: Graph,
    *,
    expected_atoms: Optional[List[str]] = None,
    expected_answers: Optional[List[str]] = None,
    min_supports_per_answer: int = 2,
) -> ValidationReport:
    """Structural + seeding validation.

    Errors (hard failures):
      - Duplicate node ids in the raw node list.
      - Any edge referencing a non-existent node.
      - Layer-ordering violation: edge sources/targets must respect
        1->2, 2->3, 3->4 (with question at layer 1).
      - Missing / duplicate question node (Layer 1 must be exactly one node id 'q').
      - Layer 4 label set != expected_answers (when provided): reports missing + extra.
      - Any expected_atoms not present in Layer 3 labels (fuzzy match on normalized text).
      - Layer 4 answers with fewer than min_supports_per_answer incoming Layer-3 edges.

    Warnings (soft):
      - Orphan Layer 2 (no children) or Layer 3 (no parent, or no child answer).
      - Extra Layer 3 nodes beyond expected_atoms (prompt allows up to 2 extras).
      - Duplicate labels within a layer.
    """
    errors: List[str] = []
    warnings: List[str] = []

    raw_nodes: List[Dict[str, Any]] = list(graph.meta.get("graph", {}).get("nodes", []))
    if raw_nodes:
        ids_seen: Dict[str, int] = {}
        for n in raw_nodes:
            ids_seen[n.get("id", "")] = ids_seen.get(n.get("id", ""), 0) + 1
        dups = [nid for nid, c in ids_seen.items() if c > 1]
        if dups:
            errors.append(f"duplicate node ids: {dups}")

    # Layer 1
    l1 = graph.by_layer("1")
    if len(l1) != 1:
        errors.append(f"expected exactly 1 Layer 1 node, got {len(l1)}")
    elif l1[0]["id"] != "q":
        warnings.append(f"Layer 1 node id is {l1[0]['id']!r}, expected 'q'")

    l2 = graph.by_layer("2")
    l3 = graph.by_layer("3")
    l4 = graph.by_layer("4")
    l2_ids = {n["id"] for n in l2}
    l3_ids = {n["id"] for n in l3}
    l4_ids = {n["id"] for n in l4}
    q_ids = {n["id"] for n in l1}

    # Edge structural checks
    layer_of = {n["id"]: str(n.get("layer", "")) for n in graph.nodes_by_id.values()}
    for e in graph.edges:
        s, t = e.get("source"), e.get("target")
        if s not in graph.nodes_by_id:
            errors.append(f"edge {s!r} -> {t!r}: source is not a node")
            continue
        if t not in graph.nodes_by_id:
            errors.append(f"edge {s!r} -> {t!r}: target is not a node")
            continue
        ls, lt = layer_of.get(s, "?"), layer_of.get(t, "?")
        expected = {("1", "2"), ("2", "3"), ("3", "4")}
        if (ls, lt) not in expected:
            errors.append(f"edge {s!r}(L{ls}) -> {t!r}(L{lt}) violates layer ordering")

    # Connectivity: children of q
    edges_from = {}
    edges_to = {}
    for e in graph.edges:
        edges_from.setdefault(e.get("source"), []).append(e.get("target"))
        edges_to.setdefault(e.get("target"), []).append(e.get("source"))

    for qid in q_ids:
        children = [t for t in edges_from.get(qid, []) if t in l2_ids]
        if not children:
            errors.append(f"Layer 1 node {qid!r} has no Layer 2 children")

    for sq in l2:
        children = [t for t in edges_from.get(sq["id"], []) if t in l3_ids]
        if not children:
            warnings.append(f"sub-question {sq['id']!r} has no Layer 3 children")

    for c in l3:
        parents = [s for s in edges_to.get(c["id"], []) if s in l2_ids]
        answers = [t for t in edges_from.get(c["id"], []) if t in l4_ids]
        if not parents:
            warnings.append(f"Layer 3 node {c['id']!r} has no sub-question parent")
        if not answers:
            warnings.append(f"Layer 3 node {c['id']!r} not connected to any Layer 4 answer")

    # Layer 4 support count
    for a in l4:
        supports = {s for s in edges_to.get(a["id"], []) if s in l3_ids}
        if len(supports) < min_supports_per_answer:
            errors.append(
                f"answer {a['id']!r} ({a.get('label','')!r}) has {len(supports)} "
                f"Layer 3 supports; need ≥ {min_supports_per_answer}"
            )

    # Duplicate labels within a layer (warn)
    for layer_name, nodes in (("L2", l2), ("L3", l3), ("L4", l4)):
        labels: Dict[str, int] = {}
        for n in nodes:
            lab = n.get("label", "")
            labels[lab] = labels.get(lab, 0) + 1
        dups = [lab for lab, c in labels.items() if c > 1]
        if dups:
            warnings.append(f"{layer_name} has duplicate labels: {dups}")

    # Expected atoms (fuzzy match on normalized labels)
    if expected_atoms is not None:
        l3_norm = {_normalize(n.get("label", "")): n["id"] for n in l3}
        missing_atoms = []
        for atom in expected_atoms:
            if _normalize(atom) not in l3_norm:
                # also check substring match — atoms can be lightly rephrased
                found = any(
                    _normalize(atom) in norm or norm in _normalize(atom)
                    for norm in l3_norm
                )
                if not found:
                    missing_atoms.append(atom)
        if missing_atoms:
            errors.append(f"missing expected atoms in Layer 3: {missing_atoms}")
        extra = len(l3) - len(expected_atoms)
        if extra > 2:
            warnings.append(
                f"Layer 3 has {extra} extra nodes beyond {len(expected_atoms)} expected "
                f"atoms (prompt allows up to 2)"
            )

    # Expected answers (verbatim per prompt — exact match required)
    if expected_answers is not None:
        l4_labels = [n.get("label", "") for n in l4]
        exp_set = list(expected_answers)
        missing = [a for a in exp_set if a not in l4_labels]
        extra = [a for a in l4_labels if a not in exp_set]
        if missing:
            errors.append(f"missing plausible answers in Layer 4: {missing}")
        if extra:
            errors.append(f"Layer 4 contains answers not in plausible_answers: {extra}")

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)


def validate_graph_file(path: Path) -> ValidationReport:
    """Load a graph.json and validate it using the atoms/answers stored in the file itself."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("graph"):
        return ValidationReport(ok=False, errors=[f"{path}: no graph in payload"])
    graph = Graph(
        nodes_by_id={n["id"]: n for n in data["graph"]["nodes"]},
        edges=list(data["graph"]["edges"]),
        meta=data,
    )
    return validate_graph(
        graph,
        expected_atoms=data.get("atoms"),
        expected_answers=data.get("plausible_answers"),
    )


# ------------------------------------------------------------------ #
# Pretty print                                                       #
# ------------------------------------------------------------------ #

def pretty_print(graph: Graph) -> str:
    """Plain-text layered rendering: Q -> subq -> Layer 3 -> answers."""
    lines = []
    q = graph.by_layer("1")
    lines.append(f"Q: {q[0]['label']}" if q else "Q: (missing)")
    edges_from = {}
    for e in graph.edges:
        edges_from.setdefault(e["source"], []).append(e["target"])

    for sq in graph.by_layer("2"):
        lines.append(f"  ├─ [{sq['id']}] {sq['label']}")
        for c_id in edges_from.get(sq["id"], []):
            c = graph.nodes_by_id.get(c_id)
            if c and str(c.get("layer")) == "3":
                lines.append(f"  │    ├─ [{c['id']}] {c['label']}")

    lines.append("  Answers (Layer 4):")
    for a in graph.answers():
        support = sorted(atoms_for_answer(graph, a["id"]))
        support_str = ", ".join(support) if support else "(no support)"
        lines.append(f"    • [{a['id']}] {a['label']}")
        lines.append(f"        supported by: {support_str}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #

def _cli() -> None:
    import argparse
    from generate_dataset import create_llm_client
    from run_lattices import load_rows, stringify_sources

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help="Base dataset (required for pipeline mode)")
    ap.add_argument("--graph", type=Path, default=None,
                    help="Path to a single graph.json (pipeline or single-validate)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Path to graphs/<run> to validate all graph.json under it")
    ap.add_argument("--qid", type=int, default=None,
                    help="Row id (required for pipeline mode)")
    ap.add_argument("--provider", choices=("openai", "vllm", "local"), default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--show-graph", action="store_true",
                    help="Also print the pretty-rendered graph")
    ap.add_argument("--validate", action="store_true",
                    help="Validate the graph (use --graph or --run-dir); skip pipeline")
    args = ap.parse_args()

    if args.validate:
        if args.run_dir:
            paths = sorted(args.run_dir.glob("*/graph.json"))
            if not paths:
                raise SystemExit(f"no graph.json under {args.run_dir}")
            fails = warns = 0
            for p in paths:
                rep = validate_graph_file(p)
                tag = "OK  " if rep.ok else "FAIL"
                extras = []
                if rep.errors:
                    extras.append(f"{len(rep.errors)}e")
                    fails += 1
                if rep.warnings:
                    extras.append(f"{len(rep.warnings)}w")
                    warns += 1
                print(f"[{tag}] {p.parent.name:>6}  " + (" ".join(extras) or ""))
                for e in rep.errors:
                    print(f"        ERROR: {e}")
                for w in rep.warnings:
                    print(f"        warn:  {w}")
            print(f"\n{len(paths)} graphs — {fails} failed, {warns} with warnings")
            return
        if args.graph:
            print(validate_graph_file(args.graph).format())
            return
        raise SystemExit("--validate needs --graph or --run-dir")

    if not (args.source and args.graph and args.qid is not None):
        raise SystemExit("pipeline mode requires --source, --graph, --qid")

    rows = {r["id"]: r for r in load_rows(args.source)}
    if args.qid not in rows:
        raise SystemExit(f"qid {args.qid} not in {args.source}")
    row = rows[args.qid]

    graph = load_graph(args.graph)
    mode = graph.meta.get("mode")
    atoms = graph.meta.get("atoms") or []

    client = create_llm_client(provider=args.provider, model=args.model)
    result = run_pipeline(
        client=client, provider=args.provider, model=args.model,
        graph_path=args.graph, question=row["query"],
        sources=stringify_sources(row["sources"]),
        atoms=atoms,
        plausible_answers=row["plausible_answers"],
        mode=mode, temperature=args.temperature,
    )

    print(f"[qid={args.qid} mode={mode}]")
    print(f"Original answer (rank 0 of {len(row['plausible_answers'])}):")
    print(f"  [{result.original_id}] {result.original_label}")
    print(f"  Supported by ({len(result.original_atoms)}):")
    for a in result.original_atoms:
        print(f"    - {a}")

    alt = result.alternative
    if alt is None:
        print("\nNo alternative Layer 4 node found.")
    else:
        print(f"\nClosest alternative (distance={alt.distance}):")
        print(f"  [{alt.alt_id}] {alt.alt_label}")
        print(f"  Add ({len(alt.added)}):    " + (", ".join(alt.added) or "(none)"))
        print(f"  Remove ({len(alt.removed)}): " + (", ".join(alt.removed) or "(none)"))

    if args.show_graph:
        print("\n--- Graph ---")
        print(pretty_print(graph))


if __name__ == "__main__":
    _cli()
