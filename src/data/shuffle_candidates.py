#!/usr/bin/env python3

import argparse
import random
import string
from pathlib import Path

import pandas as pd


LETTERS = list(string.ascii_uppercase)


def split_candidates(value):
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


def get_geneid(candidate):
    return candidate.split("::", 1)[0].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Randomly change candidate order while keeping the candidate set fixed."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Original candidate TSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV with shuffled candidate order.",
    )
    parser.add_argument(
        "--map-output",
        required=True,
        help="TSV recording old and new candidate positions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used for candidate permutation.",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        sep="\t",
    )

    required = {
        "case_id",
        "candidates",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    rng = random.Random(args.seed)

    output_rows = []
    map_rows = []

    for _, row in df.iterrows():
        candidates = split_candidates(
            row["candidates"]
        )

        if len(candidates) > len(LETTERS):
            raise ValueError(
                f"{row['case_id']}: more than 26 candidates"
            )

        indices = list(
            range(len(candidates))
        )

        rng.shuffle(indices)

        shuffled = [
            candidates[index]
            for index in indices
        ]

        output_row = row.to_dict()
        output_row["candidates"] = "|".join(
            shuffled
        )

        output_rows.append(
            output_row
        )

        for new_position, old_position in enumerate(indices):
            map_rows.append(
                {
                    "case_id": row["case_id"],
                    "new_option": LETTERS[new_position],
                    "old_option": LETTERS[old_position],
                    "new_position": new_position,
                    "old_position": old_position,
                    "geneid": get_geneid(
                        candidates[old_position]
                    ),
                }
            )

    output_path = Path(
        args.output
    )

    map_path = Path(
        args.map_output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        output_rows
    ).to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        map_rows
    ).to_csv(
        map_path,
        sep="\t",
        index=False,
    )

    print("Seed:", args.seed)
    print("Mentions:", len(df))
    print("Saved:", output_path)
    print("Saved mapping:", map_path)


if __name__ == "__main__":
    main()
