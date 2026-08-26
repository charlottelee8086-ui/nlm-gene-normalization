import csv
import json

PRED = "family_reranker_predictions_v5_llm_species.tsv"
PAIRWISE = "family_pairwise_test_v4.jsonl"

PROMPTS = "species_round2_prompts.txt"
GOLD = "species_round2_gold.tsv"

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

# load context by pmid + mention + gold ids
context_lookup = {}

with open(PAIRWISE, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        key = (
            ex["pmid"],
            ex["mention"],
            "|".join(sorted(ex["gold_gene_ids"])),
        )

        context_lookup[key] = ex["context"]

cases = []

with open(PRED, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        if row["correct"] != "0":
            continue

        if row["oracle"] != "1":
            continue

        mention = row["mention"]

        if mention not in TARGETS:
            continue

        key = (
            row["pmid"],
            mention,
            "|".join(sorted(row["gold_gene_ids"].split("|"))),
        )

        context = context_lookup.get(key, "")

        cases.append({
            "pmid": row["pmid"],
            "mention": mention,
            "gold_gene_ids": row["gold_gene_ids"],
            "pred_gid": row["pred_gid"],
            "pred_symbol": row["pred_symbol"],
            "pred_taxid": row["pred_taxid"],
            "focus_taxid": row["focus_taxid"],
            "context": context,
        })

with open(PROMPTS, "w", encoding="utf-8") as out_p, \
     open(GOLD, "w", encoding="utf-8") as out_g:

    out_g.write(
        "case_id\tpmid\tmention\tgold_gene_ids\tpred_gid\tpred_taxid\tfocus_taxid\n"
    )

    for i, ex in enumerate(cases, start=1):
        case_id = "round2_case_{}".format(i)

        prompt = """CASE_ID: {case_id}

You are helping with biomedical gene normalization.

Task:
Identify the most likely species or organism context for this gene/protein mention.

Important:
- Do NOT choose a Gene ID.
- Only infer the species from the context.
- Use explicit context clues such as mice, rats, human cells, patients, Arabidopsis, C. elegans, zebrafish, etc.
- If the context is ambiguous, answer unclear.

Options:
- human / 9606
- mouse / 10090
- rat / 10116
- zebrafish / 7955
- Arabidopsis / 3702
- worm / 6239
- other
- unclear

Mention:
{mention}

Context:
{context}

Current system prediction:
Gene ID: {pred_gid}
Symbol: {pred_symbol}
TaxID: {pred_taxid}

GNormPlus focus taxid:
{focus_taxid}

Answer format:
{case_id}    Species: <human|mouse|rat|zebrafish|Arabidopsis|worm|other|unclear>    TaxID: <tax_id or unclear>
""".format(
            case_id=case_id,
            mention=ex["mention"],
            context=ex["context"],
            pred_gid=ex["pred_gid"],
            pred_symbol=ex["pred_symbol"],
            pred_taxid=ex["pred_taxid"],
            focus_taxid=ex["focus_taxid"],
        )

        out_p.write(prompt + "\n" + "=" * 100 + "\n")

        out_g.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                case_id,
                ex["pmid"],
                ex["mention"],
                ex["gold_gene_ids"],
                ex["pred_gid"],
                ex["pred_taxid"],
                ex["focus_taxid"],
            )
        )

print("saved:", PROMPTS)
print("saved:", GOLD)
print("cases:", len(cases))
