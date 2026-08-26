import csv
from collections import Counter

FILE = "oracle_wrong_cases.tsv"

c = Counter()

with open(FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        focus = row["focus_taxid"].strip()
        pred_tax = row["pred_taxid"].strip()

        if not focus:
            c["no_focus"] += 1
        elif pred_tax == focus:
            c["pred_species_matches_focus"] += 1
        else:
            c["pred_species_differs_from_focus"] += 1

print(c)

print("\nDetailed mismatch examples:")
with open(FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    shown = 0
    for row in reader:
        focus = row["focus_taxid"].strip()
        pred_tax = row["pred_taxid"].strip()

        if focus and pred_tax != focus:
            print(
                row["mention"],
                "gold=", row["gold_gene_ids"],
                "pred=", row["pred_gid"],
                row["pred_symbol"],
                "focus=", focus,
                "pred_tax=", pred_tax,
            )
            shown += 1

        if shown >= 30:
            break
