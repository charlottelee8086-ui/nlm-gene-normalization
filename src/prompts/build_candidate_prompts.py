#!/usr/bin/env python3

import argparse
import json
import string
from pathlib import Path

import pandas as pd


LETTERS = list(string.ascii_uppercase)


def clean_text(text):
    return (
        str(text)
        .replace("\t", " ")
        .replace("\n", " ")
    )


def parse_candidates(raw_candidates):
    if pd.isna(raw_candidates):
        return []

    candidates = []

    for raw in str(raw_candidates).split("|"):
        parts = raw.split("::")

        if len(parts) != 4:
            continue

        gene_id, tax_id, species, term = parts

        candidates.append(
            {
                "gene_id": gene_id,
                "tax_id": tax_id,
                "species": species,
                "term": term,
            }
        )

    return candidates


def format_options(candidates):
    lines = []

    for i, candidate in enumerate(candidates):
        if i >= len(LETTERS):
            raise ValueError(
                "More than 26 candidates are not supported."
            )

        lines.append(
            f"{LETTERS[i]}. "
            f"GeneID: {candidate['gene_id']} | "
            f"Species: {candidate['species']} | "
            f"Matched term: {candidate['term']}"
        )

    return "\n".join(lines)


def build_dictionary_prompt(
    mention,
    context,
    candidates,
):
    options = format_options(candidates)

    return f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention, its local context, and a closed list of candidate NCBI Gene IDs, choose the most likely candidate.

Important:
- Choose exactly one option from the candidate list.
- Do not invent a new GeneID.
- Use the context to decide the correct species and gene.
- Output only the option letter.

Mention:
{mention}

Context:
{context}

Candidates:
{options}

Only output:
Answer: <option letter>
"""


def build_sapbert_prompt(
    mention,
    context,
    candidates,
):
    options = format_options(candidates)

    return f"""You are doing biomedical gene normalization.

Task:
Given a gene/protein mention, its local context, and a list of candidate NCBI Gene IDs, choose the most likely correct candidate.

Important rules:
- Choose exactly one option from the candidate list.
- Do not invent a new Gene ID.
- Use the mention, local context, species clues, aliases, and matched terms.
- Only output the option letter, for example: Answer: A

Mention:
{mention}

Context:
{context}

Candidates:
{options}

Answer:
"""


def build_prompt(
    mention,
    context,
    candidates,
    template,
):
    if template == "dictionary":
        return build_dictionary_prompt(
            mention,
            context,
            candidates,
        )

    if template == "sapbert":
        return build_sapbert_prompt(
            mention,
            context,
            candidates,
        )

    raise ValueError(
        f"Unknown prompt template: {template}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build multiple-choice prompts from candidate GeneID lists."
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Candidate TSV file.",
    )
    parser.add_argument(
        "--context-column",
        default="context",
        help="Column containing the context shown to the model.",
    )
    parser.add_argument(
        "--template",
        choices=["dictionary", "sapbert"],
        default="dictionary",
        help="Prompt wording used in the historical experiment.",
    )
    parser.add_argument(
        "--prompts-output",
        required=True,
        help="Output JSONL file containing prompts.",
    )
    parser.add_argument(
        "--gold-output",
        default=None,
        help="Optional TSV file containing gold labels.",
    )
    parser.add_argument(
        "--skipped-output",
        default=None,
        help="Optional TSV file listing skipped cases.",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.candidates,
        sep="\t",
    )

    required_columns = {
        "case_id",
        "mention",
        "gold_geneid",
        "candidates",
        args.context_column,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    prompt_path = Path(
        args.prompts_output
    )

    prompt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gold_file = None
    skipped_file = None

    if args.gold_output:
        gold_path = Path(
            args.gold_output
        )
        gold_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        gold_file = gold_path.open(
            "w",
            encoding="utf-8",
        )

        gold_file.write(
            "case_id\tgold_geneid\tcandidates\n"
        )

    if args.skipped_output:
        skipped_path = Path(
            args.skipped_output
        )
        skipped_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        skipped_file = skipped_path.open(
            "w",
            encoding="utf-8",
        )

        skipped_file.write(
            "case_id\tmention\tgold_geneid\treason\tcontext\n"
        )

    prompt_count = 0
    skipped_count = 0

    try:
        with prompt_path.open(
            "w",
            encoding="utf-8",
        ) as prompt_file:

            for _, row in df.iterrows():
                candidates = parse_candidates(
                    row["candidates"]
                )

                if not candidates:
                    skipped_count += 1

                    if skipped_file:
                        if (
                            pd.isna(row["candidates"])
                            or not str(
                                row["candidates"]
                            ).strip()
                        ):
                            reason = "empty_candidates"
                        else:
                            reason = "bad_candidate_format"

                        skipped_file.write(
                            f"{row['case_id']}\t"
                            f"{clean_text(row['mention'])}\t"
                            f"{row['gold_geneid']}\t"
                            f"{reason}\t"
                            f"{clean_text(row[args.context_column])[:500]}\n"
                        )

                    continue

                prompt = build_prompt(
                    mention=row["mention"],
                    context=row[
                        args.context_column
                    ],
                    candidates=candidates,
                    template=args.template,
                )

                prompt_file.write(
                    json.dumps(
                        {
                            "case_id": str(
                                row["case_id"]
                            ),
                            "mention": str(
                                row["mention"]
                            ),
                            "context": str(
                                row[
                                    args.context_column
                                ]
                            ),
                            "prompt": prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if gold_file:
                    gold_file.write(
                        f"{row['case_id']}\t"
                        f"{row['gold_geneid']}\t"
                        f"{row['candidates']}\n"
                    )

                prompt_count += 1

    finally:
        if gold_file:
            gold_file.close()

        if skipped_file:
            skipped_file.close()

    print("Saved prompts:", prompt_path)
    print("Prompt template:", args.template)
    print("Prompts:", prompt_count)
    print("Skipped:", skipped_count)


if __name__ == "__main__":
    main()
