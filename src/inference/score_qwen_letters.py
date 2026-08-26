#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
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


def prepare_prompt(
    tokenizer,
    prompt,
    use_chat_template,
):
    if not use_chat_template:
        return prompt

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def score_letters(
    model,
    tokenizer,
    prompt,
    max_input_length,
    target_style,
):
    if target_style == "space":
        targets = {
            letter: [" " + letter]
            for letter in LETTERS
        }

    elif target_style == "no_space":
        targets = {
            letter: [letter]
            for letter in LETTERS
        }

    elif target_style == "both":
        targets = {
            letter: [
                letter,
                " " + letter,
            ]
            for letter in LETTERS
        }

    else:
        raise ValueError(
            f"Unknown target style: {target_style}"
        )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_length,
    ).input_ids

    if not prompt_ids:
        raise ValueError(
            "Prompt is empty after tokenization."
        )

    scores = {}

    for letter, variants in targets.items():
        variant_scores = []

        for target in variants:
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
            ).input_ids

            if not target_ids:
                continue

            input_ids = (
                prompt_ids
                + target_ids
            )

            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=model.device,
            )

            attention_mask = torch.ones_like(
                input_tensor
            )

            output = model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
            )

            logits = output.logits[0]

            score = 0.0

            for position in range(
                len(prompt_ids),
                len(input_ids),
            ):
                token_id = input_ids[
                    position
                ]

                log_probs = torch.log_softmax(
                    logits[position - 1],
                    dim=-1,
                )

                score += float(
                    log_probs[token_id].item()
                )

            variant_scores.append(
                score
            )

        scores[letter] = (
            max(variant_scores)
            if variant_scores
            else float("-inf")
        )

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Score A-E answer letters with a Qwen causal language model."
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
        required=True,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--target-style",
        default="space",
        choices=[
            "space",
            "no_space",
            "both",
        ],
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    print("Model:", args.model)
    print(
        "Use chat template:",
        args.use_chat_template,
    )
    print(
        "Target style:",
        args.target_style,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

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
            prompt = prepare_prompt(
                tokenizer,
                record["prompt"],
                args.use_chat_template,
            )

            scores = score_letters(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_input_length=args.max_input_length,
                target_style=args.target_style,
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
