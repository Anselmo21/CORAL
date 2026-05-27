import json
import sys
from search.bfs import bfs_search

with open("data/base.json") as f:
    data = json.load(f)

for row in data["rows"]:
    print(f"\nQuery {row['query_id']}: {row['query']}")
    for run in row["baseline_lattice_runs"]:
        criteria = run["criteria_used"]
        rankings = run["rankings_by_combo"]
        baseline = rankings.get("", [])
        result = bfs_search(criteria, rankings)

        print(f"  Baseline ranking : {baseline}")
        if result["found"]:
            print(f"  Counterfactual   : {result['key']}")
            print(f"  CF ranking       : {result['ranking']}")
        else:
            print(f"  No counterfactual found")
        print(f"  Calls (nodes visited): {result['calls']}")

    if "--contextcite" in sys.argv:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from search.contextcite import contextcite_search

        model_name = run["model"]

        # TODO: delete, just run a super tiny model for now that can be done on Macbook Air
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"\n  Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for run in row["baseline_lattice_runs"]:
            criteria = run["criteria_used"]
            rankings = run["rankings_by_combo"]
            baseline = rankings.get("", [])

            cc_result = contextcite_search(
                criteria=criteria,
                rankings_by_combo=rankings,
                query=row["query"],
                sources=row["sources"],
                answers=run["plausible_answers"],
                model=model,
                tokenizer=tokenizer,
                num_ablations=64,
            )

            print(f"\n  [ContextCite] Baseline: {baseline}")
            print(f"  [ContextCite] Response: {cc_result['response'][:80]}...")
            print(f"  [ContextCite] Attributions:")
            for c, s in sorted(cc_result["attributions"].items(), key=lambda x: x[1]):
                print(f"    {c:30s}  {s:+.3f}")
            if cc_result["found"]:
                print(f"  [ContextCite] Counterfactual : {cc_result['key']}")
                print(f"  [ContextCite] CF ranking     : {cc_result['ranking']}")
            else:
                print(f"  [ContextCite] No counterfactual found")
            print(f"  [ContextCite] Calls: {cc_result['calls']}  (ablations: {cc_result['num_ablations']})")

    break
