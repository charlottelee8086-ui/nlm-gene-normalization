#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

LETTERS = set("ABCDE")


def extract_letter(text):
    if text is None:
        return ""

    text = str(text).strip()

    match = re.search(
        r"Answer\s*:\s*([A-E])\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    match = re.search(
        r"\b([A-E])\b",
        text,
    )

    if match:
        return match.group(1).upper()

    return ""


def load_prompts(path, limit=None):
    records = []

    with open(
        path,
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

            if (
                limit is not None
                and len(records) >= limit
            ):
                break

    return records


def build_model_input(
    tokenizer,
    prompt,
):
    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):
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

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen for the context ablation experiment."
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
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    print("Model:", args.model)

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
        args.input,
        limit=args.limit,
    )

    print("Records:", len(records))

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
            "case_id\tletter\tdecoded\n"
        )

        for i, record in enumerate(
            records,
            start=1,
        ):
            model_input = build_model_input(
                tokenizer,
                record["prompt"],
            )

            inputs = tokenizer(
                model_input,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[
                0,
                inputs["input_ids"].shape[1]:,
            ]

            decoded = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            letter = extract_letter(
                decoded
            )

            decoded_clean = (
                decoded
                .replace("\n", " ")
                .replace("\t", " ")
                .strip()
            )

            output_file.write(
                f"{record['case_id']}\t"
                f"{letter}\t"
                f"{decoded_clean}\n"
            )

            if i % 50 == 0:
                print(
                    f"Processed {i}/{len(records)}"
                )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
