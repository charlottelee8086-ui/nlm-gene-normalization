import csv

INPUT = "family_reranker_predictions_v5_llm_species.tsv"

TARGETS = {
    "MAPK",
    "NF-kappaB",
    "NF-κB",
    "WNT",
    "Wnt",
    "Wnts",
    "H3",
    "(H3",
}

with open(INPUT, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    count = 0

    for row in reader:

        mention = row["mention"]

        if mention not in TARGETS:
            continue

        if row["correct"] != "0":
            continue

        count += 1

        print("=" * 100)
        print("CASE", count)
        print("PMID:", row["pmid"])
        print("Mention:", mention)
        print("Gold:", row["gold_gene_ids"])
        print("Prediction:", row["pred_gid"], row["pred_symbol"])
