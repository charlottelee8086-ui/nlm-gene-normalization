# -*- coding: utf-8 -*-

import json
import pandas as pd

INP = "bioelqa_dev_mentions.tsv"
OUT = "bioelqa_dev_symbol_species_prompts.jsonl"
GOLD = "bioelqa_dev_symbol_species_gold.tsv"

df = pd.read_csv(INP, sep="\t")

with open(OUT, "w", encoding="utf-8") as f_out, \
     open(GOLD, "w", encoding="utf-8") as f_gold:

    f_gold.write("case_id\tdoc_id\tmention\tgold_geneid\n")

    for i, r in df.iterrows():
        case_id = f"dev_symbol_species_case_{i+1}"

        prompt = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention and its local context, predict:
1. the most likely official gene symbol
2. the most likely species of the gene mention

Allowed species:
human, mouse, rat, zebrafish, fruit fly, arabidopsis, worm, yeast, other, unclear

Important:
- Use the mention and local context.
- Species clues may include words like human, mouse, murine, rat, zebrafish, Drosophila, Arabidopsis, C. elegans, yeast, patient, cells, model organism names.
- If the species is not clear, output unclear.
- Output exactly this format:
Symbol: <official gene symbol>
Species: <one allowed species>

Mention:
{r["mention"]}

Context:
{r["context"]}

Answer:
"""

        rec = {
            "case_id": case_id,
            "mention": r["mention"],
            "context": r["context"],
            "prompt": prompt
        }

        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_gold.write(f'{case_id}\t{r["doc_id"]}\t{r["mention"]}\t{r["gold_geneid"]}\n')

print("Saved:", OUT, GOLD)
print("Prompts:", len(df))
