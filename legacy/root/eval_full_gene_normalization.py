import sys

GOLD = sys.argv[1]
PRED = sys.argv[2]

gold = {}

with open(GOLD, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")

        # expected:
        # mention_id pmid start end mention gold_gene_ids
        if len(parts) < 6:
            continue

        mention_id = parts[0]
        gold_ids = set(parts[5].split("|"))

        gold[mention_id] = gold_ids

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")

        # expected:
        # mention_id pred_gene_id
        if len(parts) < 2:
            continue

        pred[parts[0]] = parts[1]

total = 0
correct = 0
missing = 0

for mid, gids in gold.items():
    total += 1

    pgid = pred.get(mid)

    if not pgid:
        missing += 1
        continue

    if pgid in gids:
        correct += 1

print("gold mentions:", total)
print("predicted:", total - missing)
print("missing predictions:", missing)
print("correct:", correct)
print("accuracy / recall@1:", correct / total if total else 0)
print("normalization accuracy on predicted only:", correct / (total - missing) if total - missing else 0)
