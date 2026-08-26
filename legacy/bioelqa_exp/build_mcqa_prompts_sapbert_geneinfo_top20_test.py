# -*- coding: utf-8 -*-

import json
import pandas as pd

INP = "bioelqa_test_candidates_sapbert_geneinfo_top20.tsv"
OUT_PROMPTS = "bioelqa_test_mcqa_prompts_sapbert_geneinfo_top20.jsonl"
OUT_GOLD = "bioelqa_test_mcqa_gold_sapbert_geneinfo_top20.tsv"

letters = list("ABCDEFGHIJKLMNOPQRST")

df = pd.read_csv(INP, sep="\t")

with open(OUT_PROMPTS, "w", encoding="utf-8") as f_prompt, \
     open(OUT_GOLD, "w", encoding="utf-8") as f_gold:

    f_gold.write("case_id\tgold_geneid\tcandidates\n")

    n = 0
    skipped = 0

    for _, r in df.iterrows():
        case_id = f"test_sapbert_case_{n + 1}"

        cands_raw = str(r["candidates"])
        if not cands_raw or cands_raw == "nan":
            skipped += 1
            continue

        cands = cands_raw.split("|")
        option_lines = []

        for i, c in enumerate(cands):
            if i >= len(letters):
                break

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
Given a gene/protein mention, its local context, and a list of candidate NCBI Gene IDs, choose the most likely correct candidate.

Important rules:
- Choose exactly one option from the candidate list.
- Do not invent a new Gene ID.
- Use the mention, local context, species clues, aliases, and matched terms.
- Only output the option letter, for example: Answer: A

Mention:
{r["mention"]}

Context:
{r["context"]}

Candidates:
{chr(10).join(option_lines)}

Answer:
"""

        record = {
            "case_id": case_id,
            "mention": r["mention"],
            "context": r["context"],
            "prompt": prompt
        }

        f_prompt.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_gold.write(f"{case_id}\t{r['gold_geneid']}\t{r['candidates']}\n")

        n += 1

print("Saved:", OUT_PROMPTS, OUT_GOLD)
print("Prompts:", n)
print("Skipped no-candidate:", skipped)
