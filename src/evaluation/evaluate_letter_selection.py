#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


LETTERS = list("ABCDE")


def parse_geneids(value):
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

    for raw in str(value).split("|"):
        if not raw.strip():
            continue

        parts = raw.split("::", 3)

        if len(parts) != 4:
            continue

        gene_ids.append(
            parts[0]
        )

    return gene_ids


def normalize_letter(value):
    if pd.isna(value):
        return ""

    letter = str(value).strip().upper()

    if letter in LETTERS:
        return letter

    return ""


def load_predictions(path):
    df = pd.read_csv(
        path,
        sep="\t",
    )

    if "case_id" not in df.columns:
        raise ValueError(
            "Prediction file must contain case_id."
        )

    if "letter" in df.columns:
        letter_column = "letter"
    elif "answer" in df.columns:
        letter_column = "answer"
    else:
        raise ValueError(
            "Prediction file must contain letter or answer."
        )

    return {
        str(row["case_id"]): normalize_letter(
            row[letter_column]
        )
        for _, row in df.iterrows()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate letter-based candidate selection."
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="Gold TSV produced by the prompt builder.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        default=None,
    )
    parser.add_argument(
        "--label",
        default="",
    )

    args = parser.parse_args()

    gold = pd.read_csv(
        args.gold,
        sep="\t",
        keep_default_na=False,
    )

    predictions = load_predictions(
        args.predictions
    )

    total = len(gold)
    prompts = 0
    gold_in_candidates = 0
    predictions_found = 0
    valid_predictions = 0
    missing_predictions = 0
    invalid_predictions = 0
    correct = 0

    for _, row in gold.iterrows():
        case_id = str(
            row["case_id"]
        )

        prompt_written = bool(
            int(row["prompt_written"])
        )

        if prompt_written:
            prompts += 1

        gold_geneids = parse_geneids(
            row["gold_geneid"]
        )

        candidate_geneids = (
            parse_candidate_geneids(
                row["candidates"]
            )
        )

        if (
            gold_geneids
            & set(candidate_geneids)
        ):
            gold_in_candidates += 1

        if case_id not in predictions:
            if prompt_written:
                missing_predictions += 1
            continue

        predictions_found += 1
        letter = predictions[case_id]

        if letter not in LETTERS:
            invalid_predictions += 1
            continue

        index = LETTERS.index(
            letter
        )

        if index >= len(
            candidate_geneids
        ):
            invalid_predictions += 1
            continue

        valid_predictions += 1

        predicted_geneid = (
            candidate_geneids[index]
        )

        if predicted_geneid in gold_geneids:
            correct += 1

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

    print("=== Letter selection evaluation ===")
    if args.label:
        print("Setting:", args.label)

    print("Total mentions:", total)
    print("Prompts:", prompts)
    print(
        "Gold in candidates:",
        gold_in_candidates,
    )
    print(
        "Predictions found:",
        predictions_found,
    )
    print(
        "Valid predictions:",
        valid_predictions,
    )
    print(
        "Missing predictions:",
        missing_predictions,
    )
    print(
        "Invalid predictions:",
        invalid_predictions,
    )
    print("Correct:", correct)
    print()
    print(
        f"Candidate recall: "
        f"{candidate_recall:.2%}"
    )
    print(
        f"Selector accuracy: "
        f"{selector_accuracy:.2%}"
    )
    print(
        f"Final accuracy: "
        f"{final_accuracy:.2%}"
    )

    if args.summary_output:
        summary_path = Path(
            args.summary_output
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = pd.DataFrame(
            [
                {
                    "setting": args.label,
                    "total_mentions": total,
                    "prompts": prompts,
                    "gold_in_candidates": gold_in_candidates,
                    "candidate_recall": candidate_recall,
                    "predictions_found": predictions_found,
                    "valid_predictions": valid_predictions,
                    "missing_predictions": missing_predictions,
                    "invalid_predictions": invalid_predictions,
                    "correct": correct,
                    "selector_accuracy": selector_accuracy,
                    "final_accuracy": final_accuracy,
                }
            ]
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print(
            "Saved summary:",
            summary_path,
        )


if __name__ == "__main__":
    main()
