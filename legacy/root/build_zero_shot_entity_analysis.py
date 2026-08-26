import json

train_ids = set()
test_ids = set()

for fn, target in [
    ("family_pairwise_train_v4.jsonl", train_ids),
    ("family_pairwise_test_v4.jsonl", test_ids),
]:
    with open(fn) as f:
        for line in f:
            ex = json.loads(line)

            for gid in ex["gold_gene_ids"]:
                target.add(str(gid))

zero_shot = test_ids - train_ids

print("train genes:", len(train_ids))
print("test genes:", len(test_ids))
print("zero-shot genes:", len(zero_shot))
print("zero-shot ratio:", len(zero_shot)/len(test_ids))
