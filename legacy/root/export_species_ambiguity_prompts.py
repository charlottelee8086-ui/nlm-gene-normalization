import csv
import json

IN = "oracle_wrong_cases.tsv"
OUT = "species_ambiguity_prompts.jsonl"

SPECIES_MENTIONS = {
    "MCP-1",
    "CCL2",
    "HIF-1α",
    "CXCL9",
    "CXCL10",
    "AKT",
    "CD44",
    "CD14",
}

count = 0

with open(IN, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        if row["mention"] not in SPECIES_MENTIONS:
            continue

        count += 1

        prompt = """You are helping with biomedical gene normalization.

Task:
Identify the most likely species or organism context for the gene/protein mention.

Do not choose a Gene ID.
Only choose the species/tax_id from the options below.

Options:
- human / 9606
- mouse / 10090
- rat / 10116
- Arabidopsis / 3702
- worm / 6239
- other
- unclear

Mention:
{mention}

Context information:
PMID: {pmid}
Focus species predicted by GNormPlus: {focus_taxid}

The current system predicted:
Gene ID: {pred_gid}
Symbol: {pred_symbol}
TaxID: {pred_taxid}

Gold Gene ID is hidden for evaluation.

Answer format:
Species: <human|mouse|rat|Arabidopsis|worm|other|unclear>
TaxID: <tax_id or unclear>
Reason: <brief reason>
""".format(
            mention=row["mention"],
            pmid=row["pmid"],
            focus_taxid=row["focus_taxid"],
            pred_gid=row["pred_gid"],
            pred_symbol=row["pred_symbol"],
            pred_taxid=row["pred_taxid"],
        )

        out.write(json.dumps({
            "case_id": "species_case_{}".format(count),
            "pmid": row["pmid"],
            "mention": row["mention"],
            "focus_taxid": row["focus_taxid"],
            "pred_gid": row["pred_gid"],
            "pred_symbol": row["pred_symbol"],
            "pred_taxid": row["pred_taxid"],
            "gold_gene_ids": row["gold_gene_ids"],
            "prompt": prompt,
        }, ensure_ascii=False) + "\n")

print("saved:", OUT)
print("cases:", count)
