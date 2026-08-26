# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LETTERS = list("ABCDE")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@torch.no_grad()
def score_target(model, tokenizer, input_text, target_text, device, max_input_len=512):
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(device)

    labels = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    out = model(**enc, labels=labels)

    # loss is average negative log-likelihood over target tokens
    # multiply by target length to get comparable sequence score
    target_len = (labels != tokenizer.pad_token_id).sum().item()
    score = -out.loss.item() * target_len

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="t5-base")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading model:", args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    records = load_jsonl(args.input)

    if args.limit is not None:
        records = records[:args.limit]

    print("Records:", len(records))

    out_path = Path(args.output)

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("case_id\tletter\tscore_A\tscore_B\tscore_C\tscore_D\tscore_E\n")

        for i, rec in enumerate(records, start=1):
            prompt = rec["prompt"]
            case_id = rec["case_id"]

            scores = {}

            for letter in LETTERS:
                scores[letter] = score_target(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=prompt,
                    target_text=letter,
                    device=device,
                    max_input_len=args.max_input_len,
                )

            best_letter = max(scores, key=scores.get)

            fout.write(
                f"{case_id}\t{best_letter}\t"
                f"{scores['A']}\t{scores['B']}\t{scores['C']}\t{scores['D']}\t{scores['E']}\n"
            )

            if i % 100 == 0:
                print(f"Processed {i}/{len(records)}")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
