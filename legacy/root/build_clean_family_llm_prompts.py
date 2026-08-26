import csv
import json
from collections import defaultdict

PRED = "family_reranker_predictions_v5_llm_species.tsv"
PAIRWISE = "family_pairwise_test_v4.jsonl"

PROMPTS_JSONL = "clean_family_llm_prompts.jsonl"
PROMPTS_TXT = "clean_family_llm_prompts.txt"
GOLD = "clean_family_llm_gold.tsv"

TRIGGER = {
    "MAPK",
    "MAPK (p38)",
    "mitogen-activated protein kinase",
    "mitogen-activated protein kinases",
    "mitogen activated protein kinase",
    "NF-kappaB",
    "NF-κB",
    "nuclear factor (NF)-κB",
    "ERK1/2",
}

groups = defaultdict(list)

with open(PAIRWISE, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        key = (
            ex["pmid"],
            ex["mention"],
            "|".join(sorted(ex["gold_gene_ids"])),
        )
        groups[key].append(ex)

cases = []

with open(PRED, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        mention = row["mention"]

        if mention not in TRIGGER:
            continue

        if row["correct"] == "1":
            continue

        if row["oracle"] != "1":
            continue

        key = (
            row["pmid"],
            mention,
            "|".join(sorted(row["gold_gene_ids"].split("|"))),
        )

        examples = groups.get(key)

        if not examples:
            continue

        candidates = []
        for ex in examples:
            candidates.append({
                "gene_id": ex["candidate_gene_id"],
                "tax_id": ex.get("candidate_tax_id", "UNKNOWN"),
                "symbol": ex.get("candidate_symbol", "UNKNOWN"),
                "description": ex.get("candidate_description", "UNKNOWN"),
                "aliases": ex.get("candidate_aliases", "UNKNOWN"),
            })

        cases.append({
            "pmid": row["pmid"],
            "mention": mention,
            "context": examples[0]["context"],
            "gold_gene_ids": row["gold_gene_ids"],
            "current_pred_gid": row["pred_gid"],
            "current_pred_symbol": row["pred_symbol"],
            "candidates": candidates,
        })

with open(PROMPTS_JSONL, "w", encoding="utf-8") as out_jsonl, \
     open(PROMPTS_TXT, "w", encoding="utf-8") as out_txt, \
     open(GOLD, "w", encoding="utf-8") as out_gold:

    out_gold.write("case_id\tpmid\tmention\tgold_gene_ids\tcurrent_pred_gid\n")

    for i, case in enumerate(cases, start=1):
        case_id = f"auto_family_case_{i}"

        prompt = f"""CASE_ID: {case_id}

You are doing biomedical gene normalization.

Task:
Given a gene/protein mention, its local context, and a closed list of candidate NCBI Gene IDs, choose the most likely candidate.

Important:
- Choose exactly one Gene ID from the candidate list.
- Do NOT invent a new Gene ID.
- Use context clues such as pathway names, aliases, species, family members, and nearby explanations.
- If the mention is a family-level mention such as MAPK, NF-kappaB, WNT, H3, ERK1/2, or chemokines, choose the candidate most strongly supported by the local context.
- If several candidates are biologically plausible, choose the one best matching the wording in the context.

Mention:
{case["mention"]}

Context:
{case["context"]}

Current system prediction:
Gene ID: {case["current_pred_gid"]}
Symbol: {case["current_pred_symbol"]}

Candidates:
{json.dumps(case["candidates"], ensure_ascii=False, indent=2)}

Answer format:
{case_id}    GeneID: <one candidate gene_id>
"""

        out_jsonl.write(json.dumps({
            "case_id": case_id,
            "pmid": case["pmid"],
            "mention": case["mention"],
            "prompt": prompt,
            "candidate_gene_ids": [c["gene_id"] for c in case["candidates"]],
        }, ensure_ascii=False) + "\n")

        out_txt.write(prompt + "\n" + "=" * 100 + "\n")

        out_gold.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                case_id,
                case["pmid"],
                case["mention"],
                case["gold_gene_ids"],
                case["current_pred_gid"],
            )
        )

print("saved:", PROMPTS_JSONL)
print("saved:", PROMPTS_TXT)
print("saved:", GOLD)
print("cases:", len(cases))
