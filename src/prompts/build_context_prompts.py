#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


LETTERS = list("ABCDE")

CONTEXT_COLUMNS = {
    "mention": None,
    "sentence": "ctx_sentence",
    "three_sentences": "ctx_3sent",
    "window_500": "ctx_500",
    "abstract": "ctx_abstract",
    "title_abstract": "ctx_document",
}


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


def parse_gold(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def format_options(candidates):
    lines = []

    for i, candidate in enumerate(candidates):
        lines.append(
            f"{LETTERS[i]}. "
            f"GeneID: {candidate['gene_id']} | "
            f"Species: {candidate['species']} | "
            f"Matched term: {candidate['term']}"
        )

    return "\n".join(lines)


def build_prompt(
    mention,
    candidates,
    context=None,
):
    options = format_options(
        candidates
    )

    if context is None:
        return f"""You are doing biomedical gene normalization.

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
{options}

Answer:
"""

    return f"""You are doing biomedical gene normalization.

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
{context}

Candidates:
{options}

Answer:
"""


def get_case_id(row, row_index):
    if "case_id" in row.index:
        value = row["case_id"]

        if pd.notna(value) and str(value).strip():
            return str(value)

    return f"context_case_{row_index + 1}"


def main():
    parser = argparse.ArgumentParser(
        description="Build prompts for the fixed-candidate context ablation."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Candidate TSV containing all context variants.",
    )
    parser.add_argument(
        "--context",
        required=True,
        choices=list(CONTEXT_COLUMNS),
    )
    parser.add_argument(
        "--prompts-output",
        required=True,
    )
    parser.add_argument(
        "--gold-output",
        required=True,
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        sep="\t",
    )

    required = {
        "mention",
        "gold_geneid",
        "candidates",
    }

    context_column = CONTEXT_COLUMNS[
        args.context
    ]

    if context_column is not None:
        required.add(
            context_column
        )

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    prompts_path = Path(
        args.prompts_output
    )

    gold_path = Path(
        args.gold_output
    )

    prompts_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gold_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(df)
    prompts_written = 0
    gold_in_candidates_count = 0

    with prompts_path.open(
        "w",
        encoding="utf-8",
    ) as prompts_file, gold_path.open(
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
            )

            candidates = parse_candidates(
                row["candidates"],
                max_options=args.max_options,
            )

            gold = parse_gold(
                row["gold_geneid"]
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

            prompt_written = bool(
                candidates
            )

            candidate_string = "|".join(
                candidate["raw"]
                for candidate in candidates
            )

            gold_file.write(
                f"{case_id}\t"
                f"{row['gold_geneid']}\t"
                f"{int(prompt_written)}\t"
                f"{int(gold_in_candidates)}\t"
                f"{candidate_string}\n"
            )

            if not candidates:
                continue

            context = None

            if context_column is not None:
                value = row[
                    context_column
                ]

                context = (
                    ""
                    if pd.isna(value)
                    else str(value)
                )

            prompt = build_prompt(
                mention=str(
                    row["mention"]
                ),
                candidates=candidates,
                context=context,
            )

            prompts_file.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "mention": str(
                            row["mention"]
                        ),
                        "context_setting": args.context,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            prompts_written += 1

    candidate_recall = (
        gold_in_candidates_count / total
        if total
        else 0.0
    )

    print("Context:", args.context)
    print("Total mentions:", total)
    print(
        "Prompts written:",
        prompts_written,
    )
    print(
        "Gold in candidates:",
        gold_in_candidates_count,
    )
    print(
        "Candidate recall:",
        f"{candidate_recall:.2%}",
    )
    print("Saved:", prompts_path)
    print("Saved:", gold_path)


if __name__ == "__main__":
    main()
