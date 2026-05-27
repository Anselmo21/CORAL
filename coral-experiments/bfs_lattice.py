import json
from collections import deque

with open("data/base.json") as f:
    data = json.load(f)

row = data["rows"][0]
run = row["baseline_lattice_runs"][0]
criteria = run["criteria_used"]
rankings = run["rankings_by_combo"]

def node_key(subset):
    return ", ".join(sorted(subset))

def lookup(subset):
    key = node_key(subset)
    return rankings.get(key, ["(not in data)"])[0]

# BFS from full set (top of lattice), expanding downward by removing one criterion
visited = set()
queue = deque()
start = frozenset(criteria)
queue.append(start)
visited.add(start)

print(f"Query: {row['query']}\n")
print(f"{'Level':<6} {'Node (criteria subset)':<55} {'Top-ranked answer'}")
print("-" * 110)

while queue:
    node = queue.popleft()
    level = len(node)
    label = "{" + node_key(node) + "}"
    top = lookup(node)
    print(f"{level:<6} {label:<55} {top}")

    for c in node:
        neighbor = node - {c}
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

print(f"\nTotal nodes visited: {len(visited)}")
