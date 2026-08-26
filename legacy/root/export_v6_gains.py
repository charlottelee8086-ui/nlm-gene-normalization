import csv

with open("family_reranker_predictions_v5_llm_species.tsv") as f1, \
     open("family_reranker_predictions_v6.tsv") as f2:

    r1 = csv.DictReader(f1, delimiter="\t")
    r2 = csv.DictReader(f2, delimiter="\t")

    gains = []

    for a, b in zip(r1, r2):

        gold = set(a["gold_gene_ids"].split("|"))

        v5_ok = a["pred_gid"] in gold
        v6_ok = b["v6_pred_gid"] in gold

        if (not v5_ok) and v6_ok:

            gains.append(
                (
                    a["pmid"],
                    a["mention"],
                    a["pred_gid"],
                    b["v6_pred_gid"],
                    a["gold_gene_ids"],
                )
            )

print("gain cases:", len(gains))
print()

for x in gains:
    print("\t".join(x))
