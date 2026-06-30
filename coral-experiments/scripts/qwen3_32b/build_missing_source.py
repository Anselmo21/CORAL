"""Build the run_lattices source for the curated MSMARCO qids missing from Qwen.

Reads the curated Llama dataset and writes the 10 qids absent from the Qwen source as a
{rows: [...]} file for coral-data-gen's run_lattices, reusing each query's criteria and
sources so only the backbone changes when Qwen answers them.

Writes coral-data-gen/datasets/missing-criteria.json.
"""
from __future__ import annotations

import json
from pathlib import Path

MISSING = ["1151893", "1151965", "281584", "51609", "51930",
           "52065", "52069", "52347", "723521", "723551"]

WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "coral-experiments" / "data" / "msmarco_criteria.json"
OUT = WORKSPACE / "coral-data-gen" / "datasets" / "missing-criteria.json"


def main() -> None:
    lats = json.loads(SRC.read_text())["lattices"]
    rows = [{
        "query_id": int(qid),
        "query": lats[qid]["question"],
        "sources": lats[qid].get("sources") or "",
        "criteria": list(lats[qid]["criteria"]),
        "plausible_answers": list(lats[qid].get("plausible_answers") or []),
    } for qid in MISSING]
    OUT.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
