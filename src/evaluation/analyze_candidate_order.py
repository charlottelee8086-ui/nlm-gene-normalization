#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_candidates(value):
    if pd.isna(value):
        return []

    result = []

    for item in str(value).split("|"):
        item = item.strip()

        if not item:
            continue

        result.append(
            item.split("::", 1)[0].strip()
        )

    return result


def parse_gold(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def parse_letter(value):
    value = str(value).strip()

    match = re.search(
        r"Answer\s*:\s*([A-Z])",
        value,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    match = re.search(
        r"\b([A-Z])\b",
        value,
    )

    if match:
        return match.group(1).upper()

    return ""


def load_predictions(path):
    predictions = {}

    with open(
        path,
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.rstrip("\n").split(
                "\t",
                1,
            )

            if len(parts) != 2:
                continue

            case_id, value = parts

            if case_id == "case_id":
                continue

            predictions[str(case_id)] = (
                parse_letter(value)
            )

    return predictions


def selected_geneid(letter, candidates):
    if letter not in LETTERS:
        return None

    index = LETTERS.index(letter)

    if index >= len(candidates):
        return None

    return candidates[index]


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and shuffled candidate-order predictions."
    )

    parser.add_argument(
        "--original-candidates",
        required=True,
    )
    parser.add_argument(
        "--shuffled-candidates",
        required=True,
    )
    parser.add_argument(
        "--original-predictions",
        required=True,
    )
    parser.add_argument(
        "--shuffled-predictions",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        default=None,
    )

    args = parser.parse_args()

    original = pd.read_csv(
        args.original_candidates,
        sep="\t",
    )

    shuffled = pd.read_csv(
        args.shuffled_candidates,
        sep="\t",
    )

    original = original.set_index(
        "case_id",
        drop=False,
    )

    shuffled = shuffled.set_index(
        "case_id",
        drop=False,
    )

    if set(original.index) != set(shuffled.index):
        raise ValueError(
            "Original and shuffled files do not contain the same case IDs."
        )

    original_predictions = load_predictions(
        args.original_predictions
    )

    shuffled_predictions = load_predictions(
        args.shuffled_predictions
    )

    total = len(original)

    gold_in_candidates = 0
    correct = 0

    comparable_predictions = 0
    same_geneid = 0
    same_letter = 0

    moved_cases = 0
    followed_geneid = 0
    repeated_old_letter = 0

    for case_id, row in original.iterrows():
        shuffled_row = shuffled.loc[
            case_id
        ]

        original_candidates = (
            parse_candidates(
                row["candidates"]
            )
        )

        shuffled_candidates = (
            parse_candidates(
                shuffled_row["candidates"]
            )
        )

        if sorted(original_candidates) != sorted(
            shuffled_candidates
        ):
            raise ValueError(
                f"Candidate set changed for {case_id}"
            )

        gold = parse_gold(
            row["gold_geneid"]
        )

        if gold & set(original_candidates):
            gold_in_candidates += 1

        original_letter = (
            original_predictions.get(
                str(case_id),
                "",
            )
        )

        shuffled_letter = (
            shuffled_predictions.get(
                str(case_id),
                "",
            )
        )

        original_geneid = selected_geneid(
            original_letter,
            original_candidates,
        )

        shuffled_geneid = selected_geneid(
            shuffled_letter,
            shuffled_candidates,
        )

        if (
            shuffled_geneid is not None
            and shuffled_geneid in gold
        ):
            correct += 1

        # Stability measures only use cases where both
        # runs produced a valid option.
        if (
            original_geneid is None
            or shuffled_geneid is None
        ):
            continue

        comparable_predictions += 1

        if original_geneid == shuffled_geneid:
            same_geneid += 1

        if original_letter == shuffled_letter:
            same_letter += 1

        old_position = original_candidates.index(
            original_geneid
        )

        new_position = shuffled_candidates.index(
            original_geneid
        )

        if old_position != new_position:
            moved_cases += 1

            if shuffled_geneid == original_geneid:
                followed_geneid += 1

            if shuffled_letter == original_letter:
                repeated_old_letter += 1

    recall = (
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

    same_geneid_rate = (
        same_geneid / comparable_predictions
        if comparable_predictions
        else 0.0
    )

    same_letter_rate = (
        same_letter / comparable_predictions
        if comparable_predictions
        else 0.0
    )

    followed_geneid_rate = (
        followed_geneid / moved_cases
        if moved_cases
        else 0.0
    )

    repeated_letter_rate = (
        repeated_old_letter / moved_cases
        if moved_cases
        else 0.0
    )

    print("=== Candidate-order analysis ===")
    print("Seed:", args.seed)
    print("Total mentions:", total)
    print(
        "Gold in candidates:",
        gold_in_candidates,
    )
    print("Correct:", correct)
    print(
        f"Candidate recall: {recall:.2%}"
    )
    print(
        f"Selector accuracy: {selector_accuracy:.2%}"
    )
    print(
        f"Final accuracy: {final_accuracy:.2%}"
    )
    print()
    print(
        "Comparable valid predictions:",
        comparable_predictions,
    )
    print(
        f"Same GeneID: {same_geneid_rate:.2%}"
    )
    print(
        f"Same letter: {same_letter_rate:.2%}"
    )
    print()
    print(
        "Cases where the original selected GeneID moved:",
        moved_cases,
    )
    print(
        "Followed the same GeneID:",
        f"{followed_geneid_rate:.2%}",
    )
    print(
        "Repeated the old answer letter:",
        f"{repeated_letter_rate:.2%}",
    )

    if args.summary_output:
        output_path = Path(
            args.summary_output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            [
                {
                    "seed": args.seed,
                    "total_mentions": total,
                    "gold_in_candidates": gold_in_candidates,
                    "candidate_recall": recall,
                    "correct": correct,
                    "selector_accuracy": selector_accuracy,
                    "final_accuracy": final_accuracy,
                    "comparable_predictions": comparable_predictions,
                    "same_geneid": same_geneid_rate,
                    "same_letter": same_letter_rate,
                    "moved_cases": moved_cases,
                    "followed_same_geneid": followed_geneid_rate,
                    "repeated_original_letter": repeated_letter_rate,
                }
            ]
        ).to_csv(
            output_path,
            index=False,
        )

        print(
            "\nSaved summary:",
            output_path,
        )


if __name__ == "__main__":
    main()
