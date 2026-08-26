import csv

IN = "family_reranker_predictions_v4_species.tsv"
OUT = "oracle_wrong_cases.tsv"

n = 0

with open(IN, encoding="utf-8") as f, \
     open(OUT, "w", encoding="utf-8", newline="") as out:

    reader = csv.DictReader(f, delimiter="\t")

    writer = csv.writer(out, delimiter="\t")

    writer.writerow([
        "pmid",
        "mention",
        "focus_taxid",
        "pred_gid",
        "pred_symbol",
        "pred_taxid",
        "gold_gene_ids",
        "neural_score",
        "lexical_bonus",
        "species_bonus",
        "final_score",
    ])

    for row in reader:

        if row["correct"] == "0" and row["oracle"] == "1":

            writer.writerow([
                row["pmid"],
                row["mention"],
                row["focus_taxid"],
                row["pred_gid"],
                row["pred_symbol"],
                row["pred_taxid"],
                row["gold_gene_ids"],
                row["neural_score"],
                row["lexical_bonus"],
                row["species_bonus"],
                row["final_score"],
            ])

            n += 1

print("saved:", OUT)
print("oracle_wrong:", n)
