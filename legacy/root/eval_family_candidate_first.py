import json

FILE = "family_rerank_candidates.jsonl"

total = 0
correct = 0
oracle = 0

with open(FILE, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        gold = set(ex["gold_gene_ids"])
        cands = ex["candidate_gene_ids"]

        if not cands:
            continue

        total += 1

        if cands[0] in gold:
            correct += 1

        if gold & set(cands):
            oracle += 1

print("total:", total)
print("first-candidate correct:", correct)
print("first-candidate acc:", correct / total)
print("oracle covered:", oracle)
print("oracle acc:", oracle / total)

