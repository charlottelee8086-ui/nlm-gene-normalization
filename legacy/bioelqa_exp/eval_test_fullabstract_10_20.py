# -*- coding: utf-8 -*-

import re
import string
from collections import Counter

import pandas as pd


LETTERS = list(string.ascii_uppercase)

JOBS = [
    (
        10,
        "bioelqa_test_candidates_top10_with_contexts.tsv",
        "bioelqa_test_mcqa_predictions_qwen14_fullabstract_top10.txt",
        1545,
    ),
    (
        20,
        "bioelqa_test_candidates_top20_with_contexts.tsv",
        "bioelqa_test_mcqa_predictions_qwen14_fullabstract_top20.txt",
        1562,
    ),
]


def read_predictions(path):
    pred = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t", 1)
            cid = parts[0]

            if len(parts) < 2:
                pred[cid] = "NONE"
                continue

            m = re.search(r"Answer:\s*([A-Z])", parts[1], re.I)

            if m:
                pred[cid] = m.group(1).upper()
            else:
                pred[cid] = "NONE"

    return pred


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

        out.append(parts[0])

    return out


for k, candidate_file, prediction_file, expected_gold in JOBS:

    df = pd.read_csv(candidate_file, sep="\t")
    pred = read_predictions(prediction_file)

    total = len(df)
    no_candidate = 0
    prompts = 0
    predictions_found = 0
    valid_predictions = 0
    missing_predictions = 0
    gold_in_candidates = 0
    correct = 0

    invalid = Counter()

    for _, r in df.iterrows():

        cid = str(r["case_id"])
        golds = set(str(r["gold_geneid"]).split("|"))
        cands = parse_candidates(r["candidates"], k)

        if not cands:
            no_candidate += 1
            continue

        prompts += 1

        if golds.intersection(set(cands)):
            gold_in_candidates += 1

        ans = pred.get(cid)

        if ans is None:
            missing_predictions += 1
            continue

        predictions_found += 1

        if ans not in LETTERS:
            invalid[ans] += 1
            continue

        idx = LETTERS.index(ans)

        if idx >= len(cands):
            invalid[ans] += 1
            continue

        valid_predictions += 1

        chosen = cands[idx]

        if chosen in golds:
            correct += 1

    recall = gold_in_candidates / total if total else 0
    selector = correct / gold_in_candidates if gold_in_candidates else 0
    final = correct / total if total else 0

    print("\n" + "=" * 80)
    print(f"DICTIONARY + QWEN | FULL ABSTRACT | TEST | K={k}")
    print("=" * 80)

    print("Total mentions:             ", total)
    print("No candidate:               ", no_candidate)
    print("Prompts expected:           ", prompts)
    print("Predictions found:          ", predictions_found)
    print("Missing predictions:        ", missing_predictions)
    print("Valid option predictions:   ", valid_predictions)

    print()
    print("Gold in candidates:         ", gold_in_candidates)
    print(f"Recall@{k}:                  {recall:.2%}")

    print("Correct predictions:        ", correct)
    print(f"Selector accuracy:          {selector:.2%}")
    print(f"Final accuracy:             {final:.2%}")

    print("\nInvalid answers:")
    print(dict(invalid))

    if total != 2729:
        print("\nWARNING: total is not 2729!")

    if gold_in_candidates != expected_gold:
        print(
            f"\nWARNING: expected {expected_gold} gold candidates "
            f"but found {gold_in_candidates}."
        )
        print("Candidate retrieval may have changed.")
    else:
        print("\nCandidate-retrieval sanity check: PASSED")
