import csv

FILE = "family_reranker_predictions_v5_llm_species.tsv"

TARGETS = {
    "MAPK",
    "MAPK (p38)",
    "NF-kappaB",
    "NF-κB",
    "HIF-1α",
    "CXCL9",
    "CXCL10",
    "MCP-1",
    "CCL2",
    "AKT",
    "CD14",
    "CD44",
}

count = 0

with open(FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        if row["correct"] != "0":
            continue

        mention = row["mention"]

        if mention not in TARGETS:
            continue

        count += 1

        print(
            row["pmid"],
            mention,
            row["gold_gene_ids"],
            row["pred_gid"],
            sep="\t"
        )

print()
print("total:", count)
