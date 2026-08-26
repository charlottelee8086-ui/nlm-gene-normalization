#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


SPECIES_NAMES = {
    "9606": "human",
    "10090": "mouse",
    "10116": "rat",
    "7955": "zebrafish",
    "7227": "fruit fly",
    "3702": "arabidopsis",
    "6239": "worm",
    "4932": "yeast",
}

# This follows the ranking used in the original dictionary experiments.
# Human, mouse, and rat entries are placed first. Other species follow.
SPECIES_PRIORITY = {
    "9606": 0,
    "10090": 1,
    "10116": 2,
}


def normalize_term(text):
    """Normalize a mention or alias before exact dictionary lookup."""
    return (
        str(text)
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .strip()
    )


def species_name(tax_id):
    return SPECIES_NAMES.get(str(tax_id), f"taxid:{tax_id}")


def load_dictionary(path):
    """Load the NCBI symbol/synonym lookup table."""
    alias_to_entries = defaultdict(list)

    with open(path, encoding="utf-8", errors="ignore") as f:
        header = next(f, None)

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            term, tax_id, gene_id, count = parts[:4]
            key = normalize_term(term)

            if not key:
                continue

            try:
                count = int(count)
            except ValueError:
                count = 0

            alias_to_entries[key].append(
                {
                    "gene_id": str(gene_id),
                    "tax_id": str(tax_id),
                    "term": str(term),
                    "count": count,
                }
            )

    return alias_to_entries


def get_candidates(mention, alias_to_entries, k):
    """Return up to k distinct GeneIDs for one mention."""
    key = normalize_term(mention)
    entries = alias_to_entries.get(key, [])

    entries = sorted(
        entries,
        key=lambda x: (
            SPECIES_PRIORITY.get(x["tax_id"], 99),
            -x["count"],
            x["gene_id"],
        ),
    )

    candidates = []
    seen_gene_ids = set()

    for entry in entries:
        gene_id = entry["gene_id"]

        if gene_id in seen_gene_ids:
            continue

        seen_gene_ids.add(gene_id)

        candidates.append(
            "::".join(
                [
                    gene_id,
                    entry["tax_id"],
                    species_name(entry["tax_id"]),
                    entry["term"],
                ]
            )
        )

        if len(candidates) >= k:
            break

    return candidates


def parse_gold_geneids(value):
    if pd.isna(value):
        return set()

    return {
        x.strip()
        for x in str(value).split("|")
        if x.strip()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate dictionary candidates for NLM-Gene mentions."
    )

    parser.add_argument(
        "--mentions",
        required=True,
        help="TSV file containing a mention column.",
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        help="NCBI symbol/synonym TaxID lookup table.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Maximum number of candidate GeneIDs per mention.",
    )
    parser.add_argument(
        "--case-prefix",
        default="case",
        help="Prefix used when the input file has no case_id column.",
    )

    args = parser.parse_args()

    mention_path = Path(args.mentions)
    dictionary_path = Path(args.dictionary)
    output_path = Path(args.output)

    mentions = pd.read_csv(mention_path, sep="\t")

    if "mention" not in mentions.columns:
        raise ValueError("Input file must contain a 'mention' column.")

    alias_to_entries = load_dictionary(dictionary_path)

    rows = []
    no_candidate = 0
    gold_in_candidates = 0
    has_gold = "gold_geneid" in mentions.columns

    for i, row in mentions.iterrows():
        candidates = get_candidates(
            mention=row["mention"],
            alias_to_entries=alias_to_entries,
            k=args.k,
        )

        if not candidates:
            no_candidate += 1

        output_row = row.to_dict()
        output_row["candidates"] = "|".join(candidates)

        if has_gold:
            gold = parse_gold_geneids(row["gold_geneid"])
            candidate_geneids = {
                candidate.split("::", 1)[0]
                for candidate in candidates
            }

            hit = bool(gold & candidate_geneids)
            output_row["gold_in_candidates"] = int(hit)

            if hit:
                gold_in_candidates += 1

        rows.append(output_row)

    result = pd.DataFrame(rows)
    result.to_csv(output_path, sep="\t", index=False)

    total = len(result)

    print(f"Saved: {output_path}")
    print(f"Total mentions: {total}")
    print(f"No candidate: {no_candidate} ({no_candidate / total:.2%})")

    if has_gold:
        print(
            f"Recall@{args.k}: "
            f"{gold_in_candidates}/{total} "
            f"({gold_in_candidates / total:.2%})"
        )


if __name__ == "__main__":
    main()
