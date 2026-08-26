import json

train_ids = set()

with open("family_pairwise_train_v4.jsonl") as f:
    for line in f:
        ex = json.loads(line)

        for gid in ex["gold_gene_ids"]:
            train_ids.add(str(gid))

total = 0
zero = 0

with open("family_pairwise_test_v4.jsonl") as f:
    for line in f:
        ex = json.loads(line)

        gold = {str(x) for x in ex["gold_gene_ids"]}

        total += 1

        if all(g not in train_ids for g in gold):
            zero += 1

print("mentions:", total)
print("zero-shot mentions:", zero)
print("ratio:", zero/total)
