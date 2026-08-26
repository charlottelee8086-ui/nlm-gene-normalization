#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "doc_id",
    "mention",
    "start",
    "end",
    "gold_geneid",
]

CONTEXT_COLUMNS = [
    "ctx_mention",
    "ctx_sentence",
    "ctx_3sent",
    "ctx_500",
    "ctx_abstract",
    "ctx_document",
]


def main():
    parser = argparse.ArgumentParser(
        description="Attach text context variants to a candidate file."
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Candidate TSV file.",
    )
    parser.add_argument(
        "--contexts",
        required=True,
        help="TSV file containing context variants.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV file.",
    )

    args = parser.parse_args()

    candidates = pd.read_csv(
        args.candidates,
        sep="\t",
    )

    contexts = pd.read_csv(
        args.contexts,
        sep="\t",
    )

    for column in KEY_COLUMNS:
        if column not in candidates.columns:
            raise ValueError(
                f"Candidate file is missing column: {column}"
            )

        if column not in contexts.columns:
            raise ValueError(
                f"Context file is missing column: {column}"
            )

    for column in CONTEXT_COLUMNS:
        if column not in contexts.columns:
            raise ValueError(
                f"Context file is missing column: {column}"
            )

    candidates = candidates.copy()
    candidates["_original_order"] = range(
        len(candidates)
    )

    context_table = contexts[
        KEY_COLUMNS + CONTEXT_COLUMNS
    ].copy()

    merged = candidates.merge(
        context_table,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_context"),
    )

    merged = (
        merged
        .sort_values("_original_order")
        .drop(columns=["_original_order"])
    )

    if len(merged) != len(candidates):
        raise ValueError(
            "The number of rows changed after merging contexts. "
            "Check for duplicate mention keys."
        )

    print("Candidate rows:", len(candidates))
    print("Merged rows:", len(merged))

    print("\nMissing contexts:")

    for column in CONTEXT_COLUMNS:
        print(
            f"{column}:",
            int(merged[column].isna().sum()),
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
