import json
from collections import Counter

TRIGGER = {
    "MCP-1",
    "MAPK",
    "HIF-1α",
    "CCL2",
    "CXCL9",
    "CXCL10",
}

c = Counter()

with open("family_rerank_candidates.jsonl") as f:
    for line in f:
        ex = json.loads(line)

        m = ex["mention"]

        if m in TRIGGER:
            c[m] += 1

print("total =", sum(c.values()))

for k,v in c.most_common():
    print(v, k)
