# -*- coding: utf-8 -*-

"""
Qwen replacement for BioELQA-style no-finetuning experiment.

Input JSONL format:
{
  "case_id": "...",
  "mention": "...",
  "prompt": "mention: ... options: A. ... B. ... C. ... D. ... E. ... answer:"
}

This script keeps the same compact BioELQA-style prompt and scores
the likelihood of answer symbols A-E using a causal language model.
It does not fine-tune the model.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


LETTERS = ["A", "B", "C", "D", "E"]


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_prompt_text(tokenizer, prompt, use_chat_template=False):
    """
    For the most faithful BioELQA-style setting, keep use_chat_template=False.
    This means the model sees exactly:
        mention: ... options: A. ... B. ... answer:

    If you want an instruction-tuned Qwen chat setting, set --use_chat_template.
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


@torch.no_grad()
def score_letters_for_prompt(
    model,
    tokenizer,
    prompt,
    device,
    max_input_len=512,
    target_style="space",
):
    """
    Score answer symbols A-E as continuations of the prompt.

    target_style:
      - "space": score " A", " B", ...
      - "no_space": score "A", "B", ...
      - "both": use max(score("A"), score(" A")) for each letter
    """

    if target_style == "space":
        target_texts = {letter: [" " + letter] for letter in LETTERS}
    elif target_style == "no_space":
        target_texts = {letter: [letter] for letter in LETTERS}
    elif target_style == "both":
        target_texts = {letter: [letter, " " + letter] for letter in LETTERS}
    else:
        raise ValueError(f"Unknown target_style: {target_style}")

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_len,
    ).input_ids

    if len(prompt_ids) == 0:
        raise ValueError("Empty prompt after tokenization.")

    scores = {}

    for letter, variants in target_texts.items():
        variant_scores = []

        for target in variants:
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
            ).input_ids

            if len(target_ids) == 0:
                continue

            input_ids = prompt_ids + target_ids

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor, device=device)

            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
            logits = outputs.logits[0]

            log_prob_sum = 0.0

            # For each target token at position pos, use logits at pos-1.
            for pos in range(len(prompt_ids), len(input_ids)):
                token_id = input_ids[pos]
                log_probs = torch.log_softmax(logits[pos - 1], dim=-1)
                log_prob_sum += float(log_probs[token_id].item())

            variant_scores.append(log_prob_sum)

        if not variant_scores:
            scores[letter] = float("-inf")
        else:
            scores[letter] = max(variant_scores)

    best_letter = max(scores, key=scores.get)

    return best_letter, scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        required=True,
        help="HF model name or local path, e.g. Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument(
        "--target_style",
        default="space",
        choices=["space", "no_space", "both"],
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Optional. For faithful BioELQA-style compact prompt, do NOT use this.",
    )
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    print("=" * 80)
    print("Qwen BioELQA-style no-finetuning letter scoring")
    print("Model:", args.model_name)
    print("Input:", args.input)
    print("Output:", args.output)
    print("Use chat template:", args.use_chat_template)
    print("Target style:", args.target_style)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    records = load_jsonl(args.input)

    if args.limit is not None:
        records = records[:args.limit]

    print("Records:", len(records))

    out_path = Path(args.output)

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("case_id\tletter\tscore_A\tscore_B\tscore_C\tscore_D\tscore_E\n")

        for i, rec in enumerate(records, start=1):
            case_id = rec["case_id"]
            raw_prompt = rec["prompt"]

            prompt = build_prompt_text(
                tokenizer=tokenizer,
                prompt=raw_prompt,
                use_chat_template=args.use_chat_template,
            )

            best_letter, scores = score_letters_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_input_len=args.max_input_len,
                target_style=args.target_style,
            )

            fout.write(
                f"{case_id}\t{best_letter}\t"
                f"{scores['A']}\t{scores['B']}\t{scores['C']}\t"
                f"{scores['D']}\t{scores['E']}\n"
            )

            if i % 100 == 0:
                print(f"Processed {i}/{len(records)}")

    print("=" * 80)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
