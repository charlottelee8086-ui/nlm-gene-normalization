# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd

LETTERS = list("ABCDE")


def parse_candidates(cands_raw, max_options=5):
    out = []

    if pd.isna(cands_raw):
        return out

    for c in str(cands_raw).split("|"):
        if len(out) >= max_options:
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


def option_text(c, option_format="term_species"):
    """
    faithful to BioELQA: options should look like entity names.
    """
    if option_format == "term_species":
        return f"{c['term']} {c['species']}"
    elif option_format == "term_only":
        return f"{c['term']}"
    elif option_format == "geneid_species_term":
        return f"GeneID: {c['gene_id']} | Species: {c['species']} | Matched term: {c['term']}"
    else:
        raise ValueError(f"Unknown option_format: {option_format}")


def build_prompt(mention, candidates, option_format="term_species"):
    opts = []

    for i, c in enumerate(candidates):
        opts.append(f"{LETTERS[i]}. {option_text(c, option_format)}")

    # BioELQA-style compact prompt
    prompt = f"mention: {mention} options: {' '.join(opts)} answer:"

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_gold", required=True)
    parser.add_argument("--max_options", type=int, default=5)
    parser.add_argument("--case_prefix", default="bioelqa_t5_noft")
    parser.add_argument(
        "--option_format",
        default="term_species",
        choices=["term_species", "term_only", "geneid_species_term"],
    )
    args = parser.parse_args()

    df = pd.read_csv(args.inp, sep="\t")

    n_written = 0
    n_skipped = 0
    n_gold_in_candidates = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f_jsonl, \
         open(args.out_gold, "w", encoding="utf-8") as f_gold:

        f_gold.write("case_id\tgold_geneid\tgold_in_candidates\tcandidates\n")

        for _, row in df.iterrows():
            candidates = parse_candidates(row["candidates"], max_options=args.max_options)

            if not candidates:
                n_skipped += 1
                continue

            case_id = f"{args.case_prefix}_case_{n_written + 1}"
            mention = str(row["mention"])

            gold_geneids = set(str(row["gold_geneid"]).split("|"))
            cand_geneids = [c["gene_id"] for c in candidates]
            gold_in_candidates = any(g in cand_geneids for g in gold_geneids)

            if gold_in_candidates:
                n_gold_in_candidates += 1

            prompt = build_prompt(
                mention=mention,
                candidates=candidates,
                option_format=args.option_format,
            )

            record = {
                "case_id": case_id,
                "mention": mention,
                "prompt": prompt,
                "max_options": args.max_options,
                "option_format": args.option_format,
            }

            cand_str = "|".join([c["raw"] for c in candidates])

            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_gold.write(f"{case_id}\t{row['gold_geneid']}\t{gold_in_candidates}\t{cand_str}\n")

            n_written += 1

    print("=" * 80)
    print("Input:", args.inp)
    print("Prompts written:", n_written)
    print("Skipped:", n_skipped)
    print("Gold in top-k candidates:", n_gold_in_candidates)
    print("Candidate recall@k:", round(n_gold_in_candidates / n_written, 4) if n_written else 0)
    print("Saved JSONL:", args.out_jsonl)
    print("Saved gold:", args.out_gold)


if __name__ == "__main__":
    main()
