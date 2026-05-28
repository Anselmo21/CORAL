from search.rank import sra_search, era_search
from search.bfs import bfs_search
import json
from search.lattice import iter_search_layers

DATA_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"

with open(DATA_PATH) as f:
    data = json.load(f)

CASES = []
if "runs" in data:
    for i, run in enumerate(data["runs"]):
        CASES.append({
            "name": f"Q{run['query_id']}_run{i}",
            "criteria": run["criteria_used"],
            "rankings_by_combo": run["rankings_by_combo"],
            "start_node": run["criteria_used"],
        })
else:
    for row in data["rows"]:
        for i, run in enumerate(row["baseline_lattice_runs"]):
            CASES.append({
                "name": f"Q{row['query_id']}_run{i}",
                "criteria": run["criteria_used"],
                "rankings_by_combo": run["rankings_by_combo"],
                "start_node": run["criteria_used"],
            })

def test_bfs(case, search_direction):
    print(f"Case: {case['name']}, Search direction: {search_direction}")
    
    criteria = case["criteria"]
    rankings_by_combo = case["rankings_by_combo"]
    start_node = case["start_node"]
    
    result = bfs_search(criteria, 
                        rankings_by_combo, 
                        start_node=start_node,
                        direction=search_direction)
    print(result)
    return result

def test_sra(case, search_direction):
    print(f"Case: {case['name']}")
    
    criteria = case["criteria"]
    rankings_by_combo = case["rankings_by_combo"]
    
    # get a random start node from size-3 subsets
    start_node = case["start_node"]
    print(f"Start node: {start_node}")
    
    result = sra_search(criteria, 
                        rankings_by_combo, 
                        start_node=start_node,
                        direction=search_direction,
                        verbose=False)
    print(result)
    return result

def test_era(case, search_direction):
    print(f"Case: {case['name']}")

    criteria = case["criteria"]
    rankings_by_combo = case["rankings_by_combo"]

    budget = max(1, int(len(rankings_by_combo) * 0.2))
    result = era_search(criteria, rankings_by_combo, budget=budget, direction=search_direction, verbose=True)
    print(result)
    return result

def test_lattice_up():
    criteria = ["A", "B", "C", "D", "E"]
    start_node = ["A", "B"]
    direction = "up"
    for layer_size, list_of_subsets in iter_search_layers(criteria, start_node, direction):
        print(f"Layer size: {layer_size}")
        print(f"List of subsets: {list_of_subsets}")
        for subset in list_of_subsets:
            print(f"Subset: {subset}")
        
        if layer_size == 3: 
            assert len(list_of_subsets) == 10

def test_lattice_down():
    criteria = ["A", "B", "C", "D", "E"]
    start_node = ["A", "B", "C", "D"]
    direction = "down"
    for layer_size, list_of_subsets in iter_search_layers(criteria, start_node, direction):
        print(f"Layer size: {layer_size}")
        print(f"List of subsets: {list_of_subsets}")
        for subset in list_of_subsets:
            print(f"Subset: {subset}")

if __name__ == "__main__":
    # test_lattice_up()
    # test_lattice_down()
    search_direction = "down"
    sra_true_cfs = 0
    era_true_cfs = 0
    pruned_nodes = 0
    for case in CASES:
        print("="*100)
        res_bfs = test_bfs(case, search_direction)
        
        print("*"*70)
        res_sra = test_sra(case, search_direction)
        is_cf_found = res_bfs["size"] == res_sra["size"]
        if is_cf_found:
            sra_true_cfs += 1
        print(">>> TRUE CF FOUND: ", is_cf_found)

        print("*"*70)
        res_era = test_era(case, search_direction)
        is_cf_found = res_bfs["size"] == res_era["size"]
        if is_cf_found:
            era_true_cfs += 1
        print(">>> TRUE CF FOUND: ", is_cf_found)
        pruned_nodes += res_era["pruned_nodes"]

    print(f"SRA Total true CFs found: {sra_true_cfs} / {len(CASES)}")
    print(f"ERA Total true CFs found: {era_true_cfs} / {len(CASES)}")
    print(f"ERA Total pruned nodes: {pruned_nodes}")