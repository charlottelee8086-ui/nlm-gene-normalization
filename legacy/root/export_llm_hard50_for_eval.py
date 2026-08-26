import json
from pathlib import Path

IN = Path("llm_hard_single_gene_50.jsonl")
PROMPTS = Path("llm_hard50_prompts_only.jsonl")
GOLD = Path("llm_hard50_gold.tsv")

with open(IN, encoding="utf-8") as f, \
     open(PROMPTS, "w", encoding="utf-8") as out_p, \
     open(GOLD, "w", encoding="utf-8") as out_g:

    out_g.write("case_id\tpmid\tmention\tgold_gene_id\n")

    for i, line in enumerate(f, start=1):
        ex = json.loads(line)

        case_id = "case_{}".format(i)
        gold = ex["gold_gene_ids"][0]

        prompt = ex["prompt"]

        # remove accidental gold if any external wrapper printed it;
        # the prompt itself should not contain gold.
        out_p.write(json.dumps({
            "case_id": case_id,
            "pmid": ex["pmid"],
            "mention": ex["mention"],
            "prompt": prompt,
            "candidate_gene_ids": [c["gene_id"] for c in ex["candidates"]],
        }, ensure_ascii=False) + "\n")

        out_g.write("{}\t{}\t{}\t{}\n".format(
            case_id,
            ex["pmid"],
            ex["mention"],
            gold,
        ))

print("saved:", PROMPTS)
print("saved:", GOLD)
