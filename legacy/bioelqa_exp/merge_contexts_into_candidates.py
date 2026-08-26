# -*- coding: utf-8 -*-

import argparse
import pandas as pd

CONTEXT_COLS = [
    "ctx_mention",
    "ctx_sentence",
    "ctx_3sent",
    "ctx_500",
    "ctx_abstract",
    "ctx_document",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates, sep="\t")
    ctx = pd.read_csv(args.contexts, sep="\t")

    print("Candidate columns:")
    print(cand.columns.tolist())

    print("\nContext columns:")
    print(ctx.columns.tolist())

    key_cols = ["doc_id", "mention", "start", "end", "gold_geneid"]

    for c in key_cols:
        if c not in cand.columns:
            raise ValueError(f"Missing key column in candidates file: {c}")
        if c not in ctx.columns:
            raise ValueError(f"Missing key column in contexts file: {c}")

    for c in CONTEXT_COLS:
        if c not in ctx.columns:
            raise ValueError(f"Missing context column in contexts file: {c}")

    cand["_row_id"] = range(len(cand))

    ctx_small = ctx[key_cols + CONTEXT_COLS].copy()

    merged = cand.merge(
        ctx_small,
        on=key_cols,
        how="left",
        suffixes=("", "_ctx"),
    )

    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])

    print("\nCandidates rows:", len(cand))
    print("Merged rows:", len(merged))

    if len(merged) != len(cand):
        print("WARNING: row count changed after merge. Check duplicate keys.")

    print("\nMissing context counts:")
    for c in CONTEXT_COLS:
        print(c, merged[c].isna().sum())

    merged.to_csv(args.out, sep="\t", index=False)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
