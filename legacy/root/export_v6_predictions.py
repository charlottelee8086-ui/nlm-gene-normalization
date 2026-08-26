import csv
import re

BASE = "family_reranker_predictions_v5_llm_species.tsv"

GOLD = "family_member_llm_gold.tsv"
PRED = "family_member_llm_predictions.txt"

OUT = "family_reranker_predictions_v6.tsv"

# -------------------------
# family LLM prediction
# -------------------------

llm_pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(family_case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if m_case and m_gid:
            llm_pred[m_case.group(1)] = m_gid.group(1)

# -------------------------
# case -> prediction
# -------------------------

replacement = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)

    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 5:
            continue

        case_id = parts[0]
        pmid = parts[1]
        mention = parts[2]
        gold_gene_ids = parts[3]

        if case_id not in llm_pred:
            continue

        key = (
            pmid,
            mention,
            "|".join(sorted(gold_gene_ids.split("|")))
        )

        replacement[key] = llm_pred[case_id]

# -------------------------
# apply corrections
# -------------------------

total = 0
correct = 0

with open(BASE, encoding="utf-8") as f, \
     open(OUT, "w", encoding="utf-8") as out:

    reader = csv.DictReader(f, delimiter="\t")

    header = reader.fieldnames + ["v6_pred_gid"]
    out.write("\t".join(header) + "\n")

    for row in reader:

        key = (
            row["pmid"],
            row["mention"],
            "|".join(sorted(row["gold_gene_ids"].split("|")))
        )

        pred_gid = row["pred_gid"]

        if key in replacement:
            pred_gid = replacement[key]

        row["v6_pred_gid"] = pred_gid

        out.write(
            "\t".join(
                row[h]
                for h in header
            ) + "\n"
        )

        total += 1

        gold = set(row["gold_gene_ids"].split("|"))

        if pred_gid in gold:
            correct += 1

print("total:", total)
print("correct:", correct)
print("acc:", correct / total)
print("saved:", OUT)
