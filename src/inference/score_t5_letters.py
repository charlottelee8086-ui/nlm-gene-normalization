#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


LETTERS = list("ABCDE")


def load_prompts(path):
    records = []

    with open(
        path,
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


@torch.no_grad()
def score_target(
    model,
    tokenizer,
    prompt,
    target,
    device,
    max_input_length,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    labels = tokenizer(
        target,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    output = model(
        **inputs,
        labels=labels,
    )

    target_length = (
        labels != tokenizer.pad_token_id
    ).sum().item()

    return (
        -output.loss.item()
        * target_length
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score A-E answer letters with a T5 model."
    )

    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--model",
        default="t5-base",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model
    ).to(device)

    model.eval()

    records = load_prompts(
        args.input
    )

    if args.limit is not None:
        records = records[
            :args.limit
        ]

    output_path = Path(
        args.output
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
            "case_id\tletter\t"
            "score_A\tscore_B\tscore_C\t"
            "score_D\tscore_E\n"
        )

        for i, record in enumerate(
            records,
            start=1,
        ):
            scores = {}

            for letter in LETTERS:
                scores[letter] = score_target(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=record["prompt"],
                    target=letter,
                    device=device,
                    max_input_length=args.max_input_length,
                )

            best_letter = max(
                scores,
                key=scores.get,
            )

            output_file.write(
                f"{record['case_id']}\t"
                f"{best_letter}\t"
                f"{scores['A']}\t"
                f"{scores['B']}\t"
                f"{scores['C']}\t"
                f"{scores['D']}\t"
                f"{scores['E']}\n"
            )

            if i % 100 == 0:
                print(
                    f"Processed {i}/{len(records)}"
                )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
