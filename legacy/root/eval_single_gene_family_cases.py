import json

multi = set()

with open("family_rerank_candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        if len(ex["gold_gene_ids"]) > 1:
            key = (
                ex["pmid"],
                ex["mention"],
                tuple(sorted(ex["gold_gene_ids"]))
            )
            multi.add(key)

correct = 0
total = 0

with open("family_reranker_predictions_v4_species.tsv", encoding="utf-8") as f:
    next(f)

    for line in f:
        parts = line.rstrip("\n").split("\t")

        pmid = parts[0]
        mention = parts[1]
        gold = tuple(sorted(parts[6].split("|")))

        key = (pmid, mention, gold)

        if key in multi:
            continue

        total += 1

        if parts[11] == "1":
            correct += 1

print("single-gene family cases")
print("correct:", correct)
print("total:", total)
print("acc:", correct / total)
