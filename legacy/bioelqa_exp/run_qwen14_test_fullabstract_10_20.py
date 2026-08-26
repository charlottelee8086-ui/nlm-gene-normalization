# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

JOBS = [
    (
        10,
        Path("bioelqa_test_mcqa_prompts_fullabstract_top10.jsonl"),
        Path("bioelqa_test_mcqa_predictions_qwen14_fullabstract_top10.txt"),
    ),
    (
        20,
        Path("bioelqa_test_mcqa_prompts_fullabstract_top20.jsonl"),
        Path("bioelqa_test_mcqa_predictions_qwen14_fullabstract_top20.txt"),
    ),
]


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

if device != "cuda":
    raise RuntimeError("CUDA GPU not available. Do not run Qwen-14B on CPU.")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model ONCE...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()


def load_done(output):
    done = set()

    if output.exists():
        with open(output, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cid = line.split("\t", 1)[0].strip()
                if cid:
                    done.add(cid)

    return done


for k, input_path, output_path in JOBS:

    print("\n" + "=" * 80)
    print(f"STARTING K={k}")
    print("Input:", input_path)
    print("Output:", output_path)

    done = load_done(output_path)
    print("Already completed:", len(done))

    with open(input_path, encoding="utf-8") as f, \
         open(output_path, "a", encoding="utf-8") as out:

        for i, line in enumerate(f, start=1):

            ex = json.loads(line)
            cid = ex["case_id"]

            if cid in done:
                continue

            prompt = ex["prompt"]

            messages = [
                {"role": "user", "content": prompt}
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
                    max_new_tokens=32,
                    do_sample=False,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            m = re.search(r"Answer:\s*([A-Z])", response, re.I)

            if m:
                pred = m.group(1).upper()
            else:
                m = re.search(r"\b([A-Z])\b", response)
                pred = m.group(1).upper() if m else "NONE"

            out.write(f"{cid}\tAnswer: {pred}\n")
            out.flush()

            print(f"K={k} {cid} -> {pred}")

    print(f"K={k} finished.")

print("\n" + "=" * 80)
print("BOTH K=10 AND K=20 FINISHED")
