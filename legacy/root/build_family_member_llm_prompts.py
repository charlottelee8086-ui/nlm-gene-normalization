import csv
import json
from collections import defaultdict

PRED = "family_reranker_predictions_v5_llm_species.tsv"
PAIRWISE = "family_pairwise_test_v4.jsonl"

PROMPTS = "family_member_llm_prompts.txt"
GOLD = "family_member_llm_gold.tsv"

TARGETS = {
    "MAPK",
    "MAPK (p38)",
    "mitogen-activated protein kinase",
    "mitogen-activated protein kinases",
    "mitogen activated protein kinase",
    "NF-kappaB",
    "NF-κB",
    "nuclear factor (NF)-κB",
    "H3",
    "(H3",
    "histone",
    "WNT",
    "Wnt",
    "Wnts",
}

# collect candidate examples by pmid + mention + gold
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

hard_cases = []

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

        examples = groups.get(key)

        if not examples:
            continue

        hard_cases.append({
            "pmid": row["pmid"],
            "mention": mention,
            "context": examples[0]["context"],
            "gold_gene_ids": row["gold_gene_ids"],
            "current_pred_gid": row["pred_gid"],
            "current_pred_symbol": row["pred_symbol"],
            "candidates": examples,
        })

with open(PROMPTS, "w", encoding="utf-8") as out_p, \
     open(GOLD, "w", encoding="utf-8") as out_g:

    out_g.write("case_id\tpmid\tmention\tgold_gene_ids\tcurrent_pred_gid\n")

    for i, case in enumerate(hard_cases, start=1):
        case_id = "family_case_{}".format(i)

        candidates = []

        for ex in case["candidates"]:
            candidates.append({
                "gene_id": ex["candidate_gene_id"],
                "tax_id": ex.get("candidate_tax_id", "UNKNOWN"),
                "symbol": ex.get("candidate_symbol", "UNKNOWN"),
                "description": ex.get("candidate_description", "UNKNOWN"),
                "aliases": ex.get("candidate_aliases", "UNKNOWN"),
            })

        prompt = """CASE_ID: {case_id}

You are doing biomedical gene normalization.

Task:
Given a gene/protein mention, its local context, and a closed list of candidate NCBI Gene IDs, choose the most likely candidate.

Important:
- Choose exactly one Gene ID from the candidate list.
- Do NOT invent a new Gene ID.
- Use context clues such as pathway names, aliases, species, family members, and nearby explanations.
- If the mention is a family-level mention such as MAPK, NF-kappaB, WNT, H3, or ERK1/2, choose the candidate most strongly supported by the local context.
- If several candidates are biologically plausible, choose the one best matching the wording in the context.

Mention:
{mention}

Context:
{context}

Current system prediction:
Gene ID: {current_pred_gid}
Symbol: {current_pred_symbol}

Candidates:
{candidates}

Answer format:
{case_id}    GeneID: <one candidate gene_id>
""".format(
            case_id=case_id,
            mention=case["mention"],
            context=case["context"],
            current_pred_gid=case["current_pred_gid"],
            current_pred_symbol=case["current_pred_symbol"],
            candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        )

        out_p.write(prompt + "\n" + "=" * 100 + "\n")

        out_g.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                case_id,
                case["pmid"],
                case["mention"],
                case["gold_gene_ids"],
                case["current_pred_gid"],
            )
        )

print("saved:", PROMPTS)
print("saved:", GOLD)
print("cases:", len(hard_cases))
