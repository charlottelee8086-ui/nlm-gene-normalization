#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def parse_answer(response):
    match = re.search(r"Answer:\s*([A-Z])", response, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-Z])\b", response)

    if match:
        return match.group(1).upper()

    return "NONE"


def load_completed_cases(path):
    completed = set()

    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            first_field = line.rstrip("\n").split("\t", 1)[0]

            if first_field == "case_id":
                continue

            completed.add(first_field)

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen multiple-choice gene candidate selection."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="JSONL prompt file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="TSV prediction file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Model:", args.model)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    completed = load_completed_cases(output_path)

    new_file = not output_path.exists() or output_path.stat().st_size == 0

    with input_path.open(encoding="utf-8") as input_file, \
         output_path.open("a", encoding="utf-8") as output_file:

        if new_file:
            output_file.write("case_id\tanswer\n")
            output_file.flush()

        for line in input_file:
            if not line.strip():
                continue

            example = json.loads(line)
            case_id = str(example["case_id"])

            if case_id in completed:
                continue

            messages = [
                {
                    "role": "user",
                    "content": example["prompt"],
                }
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                text,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            generated = outputs[0][inputs["input_ids"].shape[-1]:]

            response = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()

            answer = parse_answer(response)

            output_file.write(f"{case_id}\t{answer}\n")
            output_file.flush()

            print(case_id, answer)


if __name__ == "__main__":
    main()
