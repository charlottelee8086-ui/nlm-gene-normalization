#!/usr/bin/env python3

import argparse
import re
import string
from collections import Counter

import pandas as pd


LETTERS = list(string.ascii_uppercase)


def parse_gold(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def parse_candidate_geneids(value):
    if pd.isna(value):
        return []

    gene_ids = []

    for candidate in str(value).split("|"):
        if not candidate.strip():
            continue

        parts = candidate.split("::")

        if not parts:
            continue

        gene_ids.append(parts[0])

    return gene_ids


def load_predictions(path):
    predictions = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) < 2:
                continue

            case_id, raw_answer = parts

            if case_id == "case_id":
                continue

            match = re.search(
                r"Answer:\s*([A-Z])",
                raw_answer,
                re.IGNORECASE,
            )

            if match:
                answer = match.group(1).upper()
            else:
                match = re.search(r"\b([A-Z])\b", raw_answer)
                answer = match.group(1).upper() if match else "NONE"

            predictions[case_id] = answer

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate retrieval and multiple-choice selection."
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Full candidate TSV, including no-candidate mentions.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Selector prediction TSV.",
    )
    parser.add_argument(
        "--gold-column",
        default="gold_geneid",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.candidates, sep="\t")

    required = {
        "case_id",
        args.gold_column,
        "candidates",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    predictions = load_predictions(args.predictions)

    total = len(df)
    no_candidate = 0
    gold_in_candidates = 0
    predictions_found = 0
    valid_predictions = 0
    correct = 0

    invalid_answers = Counter()
    wrong_choices = Counter()

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        gold = parse_gold(row[args.gold_column])
        candidates = parse_candidate_geneids(row["candidates"])

        if not candidates:
            no_candidate += 1

        retrievable = bool(gold & set(candidates))

        if retrievable:
            gold_in_candidates += 1

        if case_id not in predictions:
            continue

        predictions_found += 1
        answer = predictions[case_id]

        if answer not in LETTERS:
            invalid_answers[answer] += 1
            continue

        index = LETTERS.index(answer)

        if index >= len(candidates):
            invalid_answers[answer] += 1
            continue

        valid_predictions += 1
        selected_geneid = candidates[index]

        if selected_geneid in gold:
            correct += 1
        else:
            wrong_choices[
                (
                    answer,
                    selected_geneid,
                    "|".join(sorted(gold)),
                )
            ] += 1

    candidate_recall = (
        gold_in_candidates / total
        if total
        else 0.0
    )

    selector_accuracy = (
        correct / gold_in_candidates
        if gold_in_candidates
        else 0.0
    )

    final_accuracy = (
        correct / total
        if total
        else 0.0
    )

    print("=== Candidate selection evaluation ===")
    print(f"Total mentions: {total}")
    print(f"No candidate: {no_candidate}")
    print(f"Predictions found: {predictions_found}")
    print(f"Valid option predictions: {valid_predictions}")
    print(f"Gold in candidates: {gold_in_candidates}")
    print(f"Correct GeneIDs: {correct}")
    print()
    print(
        f"Candidate recall: "
        f"{candidate_recall:.4f} ({candidate_recall:.2%})"
    )
    print(
        f"Selector accuracy: "
        f"{selector_accuracy:.4f} ({selector_accuracy:.2%})"
    )
    print(
        f"Final accuracy: "
        f"{final_accuracy:.4f} ({final_accuracy:.2%})"
    )

    if invalid_answers:
        print("\nInvalid answers:")
        for answer, count in invalid_answers.most_common(20):
            print(count, answer)

    if wrong_choices:
        print("\nMost common wrong choices:")
        for (answer, selected, gold), count in wrong_choices.most_common(20):
            print(
                count,
                f"answer={answer}",
                f"selected={selected}",
                f"gold={gold}",
            )


if __name__ == "__main__":
    main()
