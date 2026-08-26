import json
from pathlib import Path

PROMPTS = Path("llm_rerank_prompts.jsonl")
PRED = Path("family_reranker_predictions_v4_species.tsv")
OUT = Path("llm_hard_single_gene_50.jsonl")

wrong = set()

with open(PRED, encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        pmid = parts[0]
        mention = parts[1]
        gold = parts[6].split("|")
        correct = parts[11]
        oracle = parts[12]

        if len(gold) == 1 and correct == "0" and oracle == "1":
            wrong.add((pmid, mention, gold[0]))

kept = 0

with open(PROMPTS, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        ex = json.loads(line)
        gold = ex["gold_gene_ids"]

        if len(gold) != 1:
            continue

        key = (ex["pmid"], ex["mention"], gold[0])

        if key not in wrong:
            continue

        out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        kept += 1

        if kept >= 50:
            break

print("saved:", OUT)
print("examples:", kept)
