import json
from pathlib import Path

IN = Path("llm_rerank_prompts.jsonl")
PROMPTS = Path("llm_trigger49_prompts_only.jsonl")
GOLD = Path("llm_trigger49_gold.tsv")

TRIGGER = {
    "MCP-1",
    "MAPK",
    "HIF-1α",
    "CCL2",
    "CXCL9",
    "CXCL10",
}

with open(IN, encoding="utf-8") as f, \
     open(PROMPTS, "w", encoding="utf-8") as out_p, \
     open(GOLD, "w", encoding="utf-8") as out_g:

    out_g.write("case_id\tpmid\tmention\tgold_gene_ids\n")

    idx = 0

    for line in f:
        ex = json.loads(line)

        if ex["mention"] not in TRIGGER:
            continue

        idx += 1
        case_id = f"case_{idx}"

        out_p.write(json.dumps({
            "case_id": case_id,
            "pmid": ex["pmid"],
            "mention": ex["mention"],
            "prompt": ex["prompt"],
            "candidate_gene_ids": [c["gene_id"] for c in ex["candidates"]],
        }, ensure_ascii=False) + "\n")

        out_g.write("{}\t{}\t{}\t{}\n".format(
            case_id,
            ex["pmid"],
            ex["mention"],
            "|".join(ex["gold_gene_ids"]),
        ))

print("saved:", PROMPTS)
print("saved:", GOLD)
print("cases:", idx)
