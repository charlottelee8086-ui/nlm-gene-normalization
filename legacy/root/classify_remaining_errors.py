import csv

FILE = "family_reranker_predictions_v5_llm_species.tsv"

species_like = 0
family_like = 0
family_set = 0

with open(FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        if row["correct"] != "0":
            continue

        mention = row["mention"]
        gold = row["gold_gene_ids"]

        if "|" in gold:
            family_set += 1

        elif mention.upper() in {
            "MAPK",
            "NF-KAPPAB",
            "NF-ΚB",
            "WNT",
            "WNTS",
            "AKT",
            "ERK1/2",
        }:
            family_like += 1

        else:
            species_like += 1

print("species_like:", species_like)
print("family_like:", family_like)
print("family_set:", family_set)
