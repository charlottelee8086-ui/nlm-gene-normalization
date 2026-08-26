# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd

LETTERS = list("ABCDEFGHIJKLMNOPQRST")


def parse_candidates(cands_raw, max_options=5):
    candidates = []

    if pd.isna(cands_raw):
        return candidates

    for c in str(cands_raw).split("|"):
        if len(candidates) >= max_options:
            break

        parts = c.split("::", 3)
        if len(parts) != 4:
            continue

        gene_id, tax_id, species, term = parts

        candidates.append({
            "gene_id": str(gene_id),
            "tax_id": str(tax_id),
            "species": str(species),
            "term": str(term),
            "raw": c,
        })

    return candidates


def build_prompt(row, candidates, context_col):
    option_lines = []

    for i, c in enumerate(candidates):
        option_lines.append(
            f"{LETTERS[i]}. GeneID: {c['gene_id']} | Species: {c['species']} | Matched term: {c['term']}"
        )

    mention = str(row["mention"])

    if context_col == "none":
        prompt = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention and a list of candidate NCBI Gene IDs, choose the most likely correct candidate.

Important rules:
- Choose exactly one option from the candidate list.
- Do not invent a new Gene ID.
- Use the mention, candidate species, and matched terms.
- Only output the option letter, for example: Answer: A

Mention:
{mention}

Candidates:
{chr(10).join(option_lines)}

Answer:
"""
    else:
        context_text = str(row.get(context_col, ""))

        prompt = f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention, its local context, and a list of candidate NCBI Gene IDs, choose the most likely correct candidate.

Important rules:
- Choose exactly one option from the candidate list.
- Do not invent a new Gene ID.
- Use the mention, local context, species clues, candidate species, and matched terms.
- Only output the option letter, for example: Answer: A

Mention:
{mention}

Context:
{context_text}

Candidates:
{chr(10).join(option_lines)}

Answer:
"""

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out_prompts", required=True)
    parser.add_argument("--out_gold", required=True)
    parser.add_argument(
        "--context_col",
        required=True,
        choices=["none", "ctx_sentence", "ctx_3sent", "ctx_500", "ctx_abstract", "ctx_document"],
    )
    parser.add_argument("--max_options", type=int, default=5)
    parser.add_argument("--case_prefix", default="method6_top5")
    args = parser.parse_args()

    df = pd.read_csv(args.inp, sep="\t")

    n_written = 0
    n_skipped_no_candidates = 0
    n_gold_in_candidates = 0

    with open(args.out_prompts, "w", encoding="utf-8") as f_prompt, \
         open(args.out_gold, "w", encoding="utf-8") as f_gold:

        f_gold.write("case_id\tgold_geneid\tgold_in_candidates\tcandidates\n")

        for _, row in df.iterrows():
            candidates = parse_candidates(row["candidates"], max_options=args.max_options)

            if not candidates:
                n_skipped_no_candidates += 1
                continue

            case_id = f"{args.case_prefix}_case_{n_written + 1}"

            gold_geneids = set(str(row["gold_geneid"]).split("|"))
            cand_geneids = [c["gene_id"] for c in candidates]
            gold_in_candidates = any(g in cand_geneids for g in gold_geneids)

            if gold_in_candidates:
                n_gold_in_candidates += 1

            prompt = build_prompt(row, candidates, args.context_col)

            record = {
                "case_id": case_id,
                "mention": str(row["mention"]),
                "gold_geneid": str(row["gold_geneid"]),
                "context_col": args.context_col,
                "max_options": args.max_options,
                "prompt": prompt,
            }

            cand_str = "|".join([c["raw"] for c in candidates])

            f_prompt.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_gold.write(f"{case_id}\t{row['gold_geneid']}\t{gold_in_candidates}\t{cand_str}\n")

            n_written += 1

    print("=" * 80)
    print("Input:", args.inp)
    print("Context:", args.context_col)
    print("Max options:", args.max_options)
    print("Prompts written:", n_written)
    print("Skipped no candidates:", n_skipped_no_candidates)
    print("Gold in top-k candidates:", n_gold_in_candidates)
    print("Candidate recall@k:", round(n_gold_in_candidates / n_written, 4) if n_written else 0)
    print("Saved prompts:", args.out_prompts)
    print("Saved gold:", args.out_gold)


if __name__ == "__main__":
    main()
