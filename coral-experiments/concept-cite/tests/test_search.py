import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import (
    combo_key,
    combinations_n,
    find_combo_key,
    search,
    search_base_row,
    search_trace,
)
from context_cite import ContextCiter, ConceptCriteriaPartitioner, CONCEPT_CRITERIA_PROMPT_TEMPLATE


def load_first_base_query():
    base_path = Path(__file__).resolve().parents[1] / "data" / "base.json"
    row = json.loads(base_path.read_text())["rows"][0]
    run = row["baseline_lattice_runs"][0]
    return row, run

def demo_first_base_query():
    """
    Run this file directly to inspect search on the first lattice query:
    python3 tests/test_search.py
    """
    row, run = load_first_base_query()
    criteria = run["criteria_used"]
    rankings_by_combo = run["rankings_by_combo"]
    full_key = find_combo_key(rankings_by_combo, criteria, criteria)
    assert full_key is not None
    original_answer = rankings_by_combo[full_key][0]

    # base.json stores lattice answers but not ConceptCite attribution scores.
    # Replace this with real scores from cc.get_attributions(...).data when available.
    partitioner = ConceptCriteriaPartitioner(
        question=row["query"],
        concepts=criteria,
        sources_block=row["sources"],
        answers=row["plausible_answers"],
    )
    cc = ContextCiter.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        context=partitioner.context,
        query="",
        partitioner=partitioner,
        prompt_template=CONCEPT_CRITERIA_PROMPT_TEMPLATE,
        criteria_baseline="empty",  # or "full", or None
        num_ablations=32,
    )
    attributions = cc.get_attributions(as_dataframe=True).data

    print(f"Query ID: {row['query_id']}")
    print(f"Query: {row['query']}")
    print(f"Criteria: {criteria}")
    print(f"Full-criteria answer: {original_answer}")
    print()

    print("First 10 visited nodes, top-down:")
    trace = search_trace(attributions, criteria, rankings_by_combo, direction="down")
    for step in trace[:10]:
        label = "COUNTERFACTUAL" if step.is_counterfactual else "same"
        print(f"- [{label}] score={step.score:.3f} key={step.key!r} top={step.top_answer!r}")

    hit = next((step for step in trace if step.is_counterfactual), None)
    print()
    print(f"First counterfactual hit: {hit}")


if __name__ == "__main__":
    demo_first_base_query()
