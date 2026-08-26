# -*- coding: utf-8 -*-

import argparse
import json
import string
import pandas as pd

LETTERS = list(string.ascii_uppercase)


def parse_candidates(raw, k):
    if pd.isna(raw) or not str(raw).strip():
        return []

    out = []

    for c in str(raw).split("|"):
        if len(out) >= k:
            break

        parts = c.split("::", 3)

        if len(parts) != 4:
            continue

        gene_id, tax_id, species, term = parts

        out.append({
            "gene_id": str(gene_id),
            "tax_id": str(tax_id),
            "species": str(species),
            "term": str(term),
            "raw": c,
        })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out_prompts", required=True)
    parser.add_argument("--out_gold", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.inp, sep="\t")

    n_prompts = 0
    n_no_candidate = 0
    n_gold = 0

    with open(args.out_prompts, "w", encoding="utf-8") as fp, \
         open(args.out_gold, "w", encoding="utf-8") as fg:

        fg.write(
            "case_id\tgold_geneid\tgold_in_candidates\tcandidates\n"
        )

        for _, r in df.iterrows():
            cands = parse_candidates(r["candidates"], args.k)

            if not cands:
                n_no_candidate += 1
                continue

            option_lines = []

            for i, c in enumerate(cands):
                option_lines.append(
                    f"{LETTERS[i]}. GeneID: {c['gene_id']} | "
                    f"Species: {c['species']} | "
                    f"Matched term: {c['term']}"
                )

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
{r['ctx_abstract']}

Candidates:
{chr(10).join(option_lines)}

Only output:
Answer: <option letter>
"""

            golds = set(str(r["gold_geneid"]).split("|"))
            cand_ids = [c["gene_id"] for c in cands]

            hit = bool(golds.intersection(set(cand_ids)))

            if hit:
                n_gold += 1

            case_id = str(r["case_id"])
            cand_str = "|".join(c["raw"] for c in cands)

            fp.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            fg.write(
                f"{case_id}\t{r['gold_geneid']}\t{int(hit)}\t{cand_str}\n"
            )

            n_prompts += 1

    print("=" * 80)
    print("K =", args.k)
    print("Total mentions:", len(df))
    print("Prompts:", n_prompts)
    print("No candidate:", n_no_candidate)
    print("Gold in candidates:", n_gold)
    print("Recall@K over ALL mentions:", n_gold / len(df))
    print("Saved:", args.out_prompts)
    print("Saved:", args.out_gold)


if __name__ == "__main__":
    main()
