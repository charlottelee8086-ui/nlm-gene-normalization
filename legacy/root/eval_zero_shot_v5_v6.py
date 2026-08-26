import json
import csv

# -------------------------
# train gene ids
# -------------------------

train_ids = set()

with open("family_pairwise_train_v4.jsonl") as f:
    for line in f:
        ex = json.loads(line)

        for gid in ex["gold_gene_ids"]:
            train_ids.add(str(gid))

# -------------------------
# identify zero-shot mentions
# -------------------------

zero_shot_keys = set()

with open("family_pairwise_test_v4.jsonl") as f:
    for line in f:
        ex = json.loads(line)

        gold = {str(x) for x in ex["gold_gene_ids"]}

        is_zero = all(g not in train_ids for g in gold)

        if is_zero:
            key = (
                ex["pmid"],
                ex["mention"],
                "|".join(sorted(gold))
            )
            zero_shot_keys.add(key)

# -------------------------
# evaluate v5
# -------------------------

v5_seen_total = 0
v5_seen_correct = 0

v5_zero_total = 0
v5_zero_correct = 0

with open("family_reranker_predictions_v5_llm_species.tsv") as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        key = (
            row["pmid"],
            row["mention"],
            "|".join(sorted(row["gold_gene_ids"].split("|")))
        )

        correct = int(row["correct"])

        if key in zero_shot_keys:
            v5_zero_total += 1
            v5_zero_correct += correct
        else:
            v5_seen_total += 1
            v5_seen_correct += correct

# -------------------------
# evaluate v6
# -------------------------

v6_seen_total = 0
v6_seen_correct = 0

v6_zero_total = 0
v6_zero_correct = 0

with open("family_reranker_predictions_v6.tsv") as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        gold = set(row["gold_gene_ids"].split("|"))

        key = (
            row["pmid"],
            row["mention"],
            "|".join(sorted(gold))
        )

        correct = int(row["v6_pred_gid"] in gold)

        if key in zero_shot_keys:
            v6_zero_total += 1
            v6_zero_correct += correct
        else:
            v6_seen_total += 1
            v6_seen_correct += correct

# -------------------------

print("ZERO-SHOT")

print(
    "v5:",
    v5_zero_correct,
    "/",
    v5_zero_total,
    "=",
    v5_zero_correct / v5_zero_total
)

print(
    "v6:",
    v6_zero_correct,
    "/",
    v6_zero_total,
    "=",
    v6_zero_correct / v6_zero_total
)

print()

print("SEEN")

print(
    "v5:",
    v5_seen_correct,
    "/",
    v5_seen_total,
    "=",
    v5_seen_correct / v5_seen_total
)

print(
    "v6:",
    v6_seen_correct,
    "/",
    v6_seen_total,
    "=",
    v6_seen_correct / v6_seen_total
)
