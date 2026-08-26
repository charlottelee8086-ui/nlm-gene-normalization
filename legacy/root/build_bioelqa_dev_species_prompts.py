# -*- coding: utf-8 -*-

import json
import pandas as pd

DEV = "bioelqa_dev_mentions.tsv"
OUT_PROMPTS = "bioelqa_dev_species_prompts.jsonl"
OUT_GOLD = "bioelqa_dev_species_gold.tsv"

df = pd.read_csv(DEV, sep="\t")

with open(OUT_PROMPTS, "w", encoding="utf-8") as f_prompt, \
     open(OUT_GOLD, "w", encoding="utf-8") as f_gold:

    f_gold.write("case_id\tdoc_id\tmention\tgold_geneid\n")

    for i, r in df.iterrows():
        case_id = f"dev_species_case_{i+1}"

        prompt = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention and its local context, infer the species of the mentioned gene/protein.

Choose exactly one label from:
human
mouse
rat
other
unclear

Mention:
{r['mention']}

Context:
{r['context']}

Only output:
Species: <human|mouse|rat|other|unclear>
"""

        f_prompt.write(json.dumps({
            "case_id": case_id,
            "prompt": prompt
        }, ensure_ascii=False) + "\n")

        f_gold.write(f"{case_id}\t{r['doc_id']}\t{r['mention']}\t{r['gold_geneid']}\n")

print("Saved:", OUT_PROMPTS, OUT_GOLD)
print("Total dev cases:", len(df))
