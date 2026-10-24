"""Builds the graph of thoughts prompt and reads the graph out of the reply

The prompt gives the LLM its answer under all the concepts and asks which concepts would have to go
for each other answer to win
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Sequence
from typing import Any

FENCED_JSON = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
LINE_COMMENT = re.compile(r'//[^\n]*')
BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)


def graph_prompt(
    sources: str,
    question: str,
    concept_set: str,
    concepts: Sequence[str],
    answers: Sequence[str],
    initial_answer: str,
) -> str:
    """Build the prompt for a four-layer graph of thoughts

    Each rival answer is supported by every concept except those that must go for it to win

    Args:
        sources: The retrieved sources of the query
        question: The query text
        concept_set: Either 'criteria' or 'pragmas' to set how the layers and edges are worded
        concepts: The concepts the graph must hold in Layer 3
        answers: The answers the graph must hold in Layer 4
        initial_answer: The answer ranked first under all the concepts

    Returns:
        The full prompt text
    """
    if concept_set == 'criteria':
        layer3_description = (
            'criteria — key answer-evaluation dimensions the answerer weighs '
            '(e.g., "average starting salary", "scholarship availability")'
        )
        concept_word = 'criterion'
        concepts_word = 'criteria'
        node_type = 'concept'
        verb = 'weigh'
        removal_step = (
            'P4) For each other answer A_i, find the set R_i of criterion seeds to REMOVE from the full atom set '
            'such that, evaluating only the remaining criteria, A_i beats A*. Then S(A_i) is the full set minus R_i.\n'
            '          - Each c in R_i must be a criterion that specifically favors A* over A_i, or specifically hurts A_i. '
            'For each removal, briefly say why in one clause.\n'
            '          - Do NOT remove criteria just because they seem irrelevant to A_i — only remove ones that actively work against A_i.'
        )
        edge_semantics = (
            'Edge semantics: c->a means "c is in S(a), the criterion subset under which a is the top pick." '
            "A* is wired to ALL c's (S(A*) = full seed set). Each alternative A_i is wired to ALL c's EXCEPT its "
            "removal set R_i, so most c's connect to most answers. The DIFFERENTIATION signal lives in the MISSING "
            'edges — c ↛ A_i means c is in R_i, a criterion that specifically favors A* over A_i.'
        )
        seed_target = 'to the a(s) it GENUINELY FAVORS (out)'
    else:
        layer3_description = (
            'source directives — filters on WHICH sources to prioritize when '
            'answering the sub-question (temporal, geographic, trustworthiness, '
            'or perspective)'
        )
        concept_word = 'directive'
        concepts_word = 'directives'
        node_type = 'directive'
        verb = 'apply'
        removal_step = (
            'P4) For each other answer A_i, find the set R_i of directive seeds to DROP from the full directive set '
            'such that, with the source pool broadened by dropping those filters (letting in sources that were '
            'previously excluded), the evidence now supports A_i over A*. Then S(A_i) is the full set minus R_i.\n'
            '          - Each d in R_i must be a directive whose filtering specifically excludes sources that would '
            'have supported A_i, or whose enforcement specifically keeps in sources that favor A*. For each drop, '
            'briefly say which sources come in / go out and why that flips the winner.\n'
            '          - Do NOT drop a directive just because it seems tangential — only drop ones whose filtering '
            'materially shapes the source pool against A_i.'
        )
        edge_semantics = (
            'Edge semantics: d->a means "d is in S(a), the directive subset under which a is the top pick given '
            'the sources those directives admit." A* is wired to ALL d\'s (S(A*) = full directive set — A* wins '
            "when every filter is enforced). Each alternative A_i is wired to ALL d's EXCEPT its removal set R_i, "
            'where R_i is the directives that must be RELAXED (source pool broadened) for A_i to win. The '
            'DIFFERENTIATION signal lives in the MISSING edges — d ↛ A_i means d is in R_i, a filter whose '
            'enforcement specifically props up A* over A_i.'
        )
        seed_target = (
            'to the a(s) whose supporting sources SURVIVE this filter (out) — '
            'i.e., every a except those that need this directive relaxed to win'
        )

    header = textwrap.dedent(f"""\
    Build a 4-layer reasoning graph for the QUESTION using the provided SOURCES to match your step by step reasoning process.
    The graph should be organized with the following layer structure.

    Layers:
    - Layer 1: the question (id "q").
    - Layer 2: distinct sub-questions (ids sq1..).
    - Layer 3: {layer3_description} (ids c1..).
    - Layer 4: the fixed plausible answers listed at the end (ids a1..).

    First think step by step in prose, then emit the JSON in a ```json fenced block at the END of your response. Do not skip the reasoning.

    THINK OUT LOUD (write these before the JSON):
      P1) Restate the question in one line; note what distinguishes the plausible answers from each other.
      P2) Group the Layer 3 seeds into 2-4 sub-questions.
      P3) S(A*) = ALL Layer 3 seeds. A* is the top-1 pick when the answerer {verb}s the full seed set, so by definition every seed is in S(A*) — every Layer 3 seed gets a c->A* edge. No removal for A*.
      {removal_step}
          - SIZE R_i TO WHATEVER IT TAKES. If a subtle counterfactual flips A_i with 1-2 removals, use that. If A* is heavily favored and the flip requires stripping many {concepts_word}, use a larger R_i. Do NOT settle for a small R_i that fails to flip.
          - A working larger R_i is more useful than a broken smaller one. Prefer smaller when it truly flips; grow the set as needed.
          - No two R_i may be identical; every R_i must be non-empty.
      P5) Sanity-check: each Layer 3 node has >=1 sq parent AND >=1 answer child; no two answers share a supporter set.

    Then emit:
    ```json
    {{ "nodes": [{{ "id": "...", "layer": "1|2|3|4", "type": "question|subq|{node_type}|answer", "label": "...", "description": "..." }}, ...],
      "edges": [{{ "source": "...", "target": "..." }}, ...] }}
    ```

    {edge_semantics}

    Rules:
    1) Edges only between adjacent layers: q->sq, sq->c, c->a. No skips.
    2) NO ORPHANS: every Layer 3 node MUST have >=1 incoming sq edge.
    3) Answer labels are short — no filler ("prevention", not "the best treatment is prevention").
    4) No two answers may share the same supporter set. Every S(A_i) besides S(A*) must be a proper subset of the full {concept_word} set (R_i non-empty). Size R_i to what the flip actually requires — no fixed budget.
    """)
    seed_list = '\n'.join(f'  {index + 1}. {concept}' for index, concept in enumerate(concepts))
    answer_list = '\n'.join(f'  {index + 1}. {answer}' for index, answer in enumerate(answers))
    # The required concepts and answers go last so the LLM reads them most recently
    tail = (
        f'\nLAYER 3 (exactly these {len(concepts)}, verbatim labels, no drops, no rename, '
        f'no additions):\n{seed_list}\n'
        f'Each MUST be wired to one sq (in) and {seed_target}.\n'
        f'\nLAYER 4 (exactly these {len(answers)} answers, verbatim labels, nothing else):\n'
        f'{answer_list}\n'
        f'\nBASELINE ANSWER A* (top-ranked when the answerer considers ALL Layer 3 '
        f'seeds together):\n  {initial_answer}\n'
    )
    return header + f'\nSOURCES:\n{sources}\n\nQUESTION:\n{question}\n' + tail + '\nOUTPUT:\n'


def parse_graph(response: str) -> dict[str, Any] | None:
    """Read the graph JSON from a reply or return None

    Several slices of the reply get tried in turn because models wrap the JSON in code fences and
    comments

    Args:
        response: The raw reply text

    Returns:
        The first decoded object with nodes and edges, or None if there is none
    """
    text = (response or '').strip()
    candidates = [match.group(1) for match in reversed(list(FENCED_JSON.finditer(text)))]
    unfenced = text
    if unfenced.startswith('```'):
        lines = unfenced.split('\n')
        unfenced = '\n'.join(lines[1:-1]).strip() if len(lines) > 2 else unfenced
        if unfenced.startswith('json'):
            unfenced = unfenced[4:].strip()
    candidates.append(unfenced)
    end = unfenced.rfind('}')
    start = unfenced.rfind('{', 0, end) if end != -1 else -1
    if start != -1 and end > start:
        candidates.append(unfenced[start : end + 1])
    for candidate in candidates:
        for attempt in (candidate, LINE_COMMENT.sub('', BLOCK_COMMENT.sub('', candidate))):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and 'nodes' in parsed and 'edges' in parsed:
                return parsed
    return None


def missing_seeds(
    graph: dict[str, Any], concepts: Sequence[str], answers: Sequence[str]
) -> str | None:
    """Describe which layer came back with too few nodes or return None

    The text goes into the follow-up prompt

    Args:
        graph: The parsed graph
        concepts: The concepts the graph must hold in Layer 3
        answers: The answers the graph must hold in Layer 4

    Returns:
        One clause per short layer joined by semicolons, or None if both layers are complete
    """
    nodes = graph.get('nodes') or []
    short = []
    for layer, seeds, name in (('3', concepts, 'seeds'), ('4', answers, 'answers')):
        present = sum(1 for node in nodes if str(node.get('layer')) == layer)
        if present < len(seeds):
            short.append(f'Layer {layer} has {present} of the {len(seeds)} {name}')
    return '; '.join(short) or None
