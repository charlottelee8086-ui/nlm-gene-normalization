# -*- coding: utf-8 -*-

import json
import string
import pandas as pd

INP = "bioelqa_dev_candidates_dict_top20_with_contexts.tsv"
OUT_PROMPTS = "bioelqa_dev_mcqa_prompts_dict_top20_fullabstract.jsonl"
OUT_GOLD = "bioelqa_dev_mcqa_gold_dict_top20_fullabstract.tsv"

df = pd.read_csv(INP, sep="\t")
letters = list(string.ascii_uppercase)

with open(OUT_PROMPTS, "w", encoding="utf-8") as f_prompt, \
     open(OUT_GOLD, "w", encoding="utf-8") as f_gold:

    f_gold.write("case_id\tgold_geneid\tcandidates\n")

    n = 0
    skipped = 0

    for _, r in df.iterrows():
        cands_raw = str(r["candidates"])
        if not cands_raw or cands_raw == "nan":
            skipped += 1
            continue

        cands = cands_raw.split("|")
        option_lines = []

        for i, c in enumerate(cands):
            parts = c.split("::")
            if len(parts) != 4:
                continue
            gene_id, tax_id, species, term = parts
            option_lines.append(
                f"{letters[i]}. GeneID: {gene_id} | Species: {species} | Matched term: {term}"
            )

        if not option_lines:
            skipped += 1
            continue

        prompt = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention, its local context, and a closed list of candidate NCBI Gene IDs, choose the most likely candidate.

Important:
- Choose exactly one option from the candidate list.
- Do not invent a new GeneID.
- Use the context to decide the correct species and gene.
- Output only the option letter.

Mention:
{r['mention']}

Context:
{r["ctx_abstract"]}

Candidates:
{chr(10).join(option_lines)}

Only output:
Answer: <option letter>
"""

        f_prompt.write(json.dumps({
            "case_id": r["case_id"],
            "prompt": prompt
        }, ensure_ascii=False) + "\n")

        f_gold.write(f"{r['case_id']}\t{r['gold_geneid']}\t{r['candidates']}\n")
        n += 1

print("Saved:", OUT_PROMPTS, OUT_GOLD)
print("Prompts:", n)
print("Skipped no-candidate:", skipped)
