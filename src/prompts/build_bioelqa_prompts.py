#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


LETTERS = list("ABCDE")


def parse_candidates(raw_candidates, max_options=5):
    if pd.isna(raw_candidates):
        return []

    candidates = []

    for raw in str(raw_candidates).split("|"):
        if len(candidates) >= max_options:
            break

        parts = raw.split("::", 3)

        if len(parts) != 4:
            continue

        gene_id, tax_id, species, term = parts

        candidates.append(
            {
                "gene_id": str(gene_id),
                "tax_id": str(tax_id),
                "species": str(species),
                "term": str(term),
                "raw": raw,
            }
        )

    return candidates


def format_option(candidate, option_format):
    if option_format == "term_species":
        return f"{candidate['term']} {candidate['species']}"

    if option_format == "term_only":
        return candidate["term"]

    if option_format == "geneid_species_term":
        return (
            f"GeneID: {candidate['gene_id']} | "
            f"Species: {candidate['species']} | "
            f"Matched term: {candidate['term']}"
        )

    raise ValueError(
        f"Unknown option format: {option_format}"
    )


def build_prompt(mention, candidates, option_format):
    options = []

    for i, candidate in enumerate(candidates):
        options.append(
            f"{LETTERS[i]}. "
            f"{format_option(candidate, option_format)}"
        )

    return (
        f"mention: {mention} "
        f"options: {' '.join(options)} "
        f"answer:"
    )


def get_case_id(row, row_index, prefix):
    if "case_id" in row.index:
        value = row["case_id"]

        if pd.notna(value) and str(value).strip():
            return str(value)

    return f"{prefix}_case_{row_index + 1}"


def parse_gold(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build compact BioELQA-style multiple-choice prompts."
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Candidate TSV file.",
    )
    parser.add_argument(
        "--prompts-output",
        required=True,
        help="Output JSONL prompt file.",
    )
    parser.add_argument(
        "--gold-output",
        required=True,
        help="Output TSV with one row for every input mention.",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--case-prefix",
        default="bioelqa",
    )
    parser.add_argument(
        "--option-format",
        default="term_species",
        choices=[
            "term_species",
            "term_only",
            "geneid_species_term",
        ],
    )
    parser.add_argument(
        "--gold-column",
        default=None,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.candidates,
        sep="\t",
    )

    if "mention" not in df.columns:
        raise ValueError(
            "Candidate file must contain a 'mention' column."
        )

    if "candidates" not in df.columns:
        raise ValueError(
            "Candidate file must contain a 'candidates' column."
        )

    gold_column = args.gold_column

    if gold_column is None:
        if "gold_geneid" in df.columns:
            gold_column = "gold_geneid"
        elif "gold_gene_ids" in df.columns:
            gold_column = "gold_gene_ids"
        else:
            raise ValueError(
                "Could not find a gold GeneID column."
            )

    prompt_path = Path(
        args.prompts_output
    )
    gold_path = Path(
        args.gold_output
    )

    prompt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    gold_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_count = 0
    gold_in_candidates_count = 0
    total = len(df)

    with prompt_path.open(
        "w",
        encoding="utf-8",
    ) as prompt_file, gold_path.open(
        "w",
        encoding="utf-8",
    ) as gold_file:

        gold_file.write(
            "case_id\tgold_geneid\tprompt_written\t"
            "gold_in_candidates\tcandidates\n"
        )

        for row_index, row in df.iterrows():
            case_id = get_case_id(
                row,
                row_index,
                args.case_prefix,
            )

            candidates = parse_candidates(
                row["candidates"],
                max_options=args.max_options,
            )

            gold = parse_gold(
                row[gold_column]
            )

            candidate_geneids = {
                candidate["gene_id"]
                for candidate in candidates
            }

            gold_in_candidates = bool(
                gold & candidate_geneids
            )

            if gold_in_candidates:
                gold_in_candidates_count += 1

            candidate_string = "|".join(
                candidate["raw"]
                for candidate in candidates
            )

            prompt_written = bool(candidates)

            gold_file.write(
                f"{case_id}\t"
                f"{row[gold_column]}\t"
                f"{int(prompt_written)}\t"
                f"{int(gold_in_candidates)}\t"
                f"{candidate_string}\n"
            )

            if not candidates:
                continue

            prompt = build_prompt(
                mention=str(row["mention"]),
                candidates=candidates,
                option_format=args.option_format,
            )

            prompt_file.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "mention": str(
                            row["mention"]
                        ),
                        "prompt": prompt,
                        "max_options": args.max_options,
                        "option_format": args.option_format,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            prompt_count += 1

    recall = (
        gold_in_candidates_count / total
        if total
        else 0.0
    )

    print("Saved prompts:", prompt_path)
    print("Saved gold:", gold_path)
    print("Total mentions:", total)
    print("Prompts written:", prompt_count)
    print(
        "Gold in candidates:",
        gold_in_candidates_count,
    )
    print(
        f"Candidate recall@{args.max_options}: "
        f"{recall:.2%}"
    )


if __name__ == "__main__":
    main()
