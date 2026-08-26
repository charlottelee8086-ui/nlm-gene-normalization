# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


LETTERS = set("ABCDE")  # top-5 only


def extract_letter(text):
    """
    Extract option letter from Qwen output.
    Accepts:
    Answer: A
    A
    The answer is A
    """
    if text is None:
        return ""

    text = str(text).strip()

    m = re.search(r"Answer\s*:\s*([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()

    return ""


def load_records(path, limit=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def build_model_input(tokenizer, prompt):
    """
    For Qwen-Instruct models, use chat template.
    If the tokenizer has no chat template, fall back to raw prompt.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True,
                        help="HF model name or local model path, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="For testing only. Example: --limit 5")
    args = parser.parse_args()

    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print("Loading model:", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    records = load_records(args.input, limit=args.limit)
    print("Records:", len(records))

    out_path = Path(args.output)

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("case_id\tletter\tdecoded\n")

        for i, rec in enumerate(records, start=1):
            case_id = rec["case_id"]
            prompt = rec["prompt"]

            model_input = build_model_input(tokenizer, prompt)

            inputs = tokenizer(
                model_input,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            letter = extract_letter(decoded)

            # keep one-line output
            decoded_clean = decoded.replace("\n", " ").replace("\t", " ").strip()

            fout.write(f"{case_id}\t{letter}\t{decoded_clean}\n")

            if i % 50 == 0:
                print(f"Processed {i}/{len(records)}")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
