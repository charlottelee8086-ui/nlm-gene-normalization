import json
from collections import Counter

c = Counter()

with open("family_rerank_candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        n = len(ex["gold_gene_ids"])

        if n == 1:
            c["single"] += 1
        else:
            c["multi"] += 1

        c[f"gold_{n}"] += 1

print("single:", c["single"])
print("multi:", c["multi"])

print()
for k in sorted(c):
    if k.startswith("gold_"):
        print(k, c[k])
