# -*- coding: utf-8 -*-

import pandas as pd

CTX = "../bioelqa_test_mentions_contexts.tsv"

JOBS = [
    (
        "bioelqa_test_candidates.tsv",
        "bioelqa_test_candidates_top10_with_contexts.tsv",
    ),
    (
        "bioelqa_test_candidates_top20.tsv",
        "bioelqa_test_candidates_top20_with_contexts.tsv",
    ),
]

CONTEXT_COLS = [
    "ctx_mention",
    "ctx_sentence",
    "ctx_3sent",
    "ctx_500",
    "ctx_abstract",
    "ctx_document",
]


def clean(v):
    return str(v).strip()


ctx = pd.read_csv(CTX, sep="\t")

print("Context rows:", len(ctx))

for inp, out in JOBS:
    cand = pd.read_csv(inp, sep="\t")

    print("\n" + "=" * 80)
    print("INPUT:", inp)
    print("Rows:", len(cand))

    if len(cand) != len(ctx):
        raise ValueError(
            f"Row-count mismatch: candidates={len(cand)}, contexts={len(ctx)}"
        )

    # Candidate files were created directly from test mentions in row order.
    # Verify this before attaching context.
    check_cols = ["doc_id", "mention", "gold_geneid"]

    for col in check_cols:
        a = cand[col].map(clean).tolist()
        b = ctx[col].map(clean).tolist()

        bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]

        if bad:
            print("Mismatch column:", col)
            print("First mismatch rows:", bad[:10])
            for i in bad[:3]:
                print("candidate:", cand.iloc[i][check_cols].to_dict())
                print("context:  ", ctx.iloc[i][check_cols].to_dict())
            raise ValueError(f"Row alignment failed for {col}")

    for col in CONTEXT_COLS:
        cand[col] = ctx[col].values

    cand.to_csv(out, sep="\t", index=False)

    print("Saved:", out)
    print("Missing abstracts:", cand["ctx_abstract"].isna().sum())
    print(
        "Empty abstracts:",
        (cand["ctx_abstract"].fillna("").astype(str).str.len() == 0).sum(),
    )
    print(
        "First abstract length:",
        len(str(cand.iloc[0]["ctx_abstract"])),
    )

print("\nAll context attachments completed.")
