# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

INPUT = Path("bioelqa_dev_mcqa_prompts_top20.jsonl")
OUTPUT = Path("bioelqa_dev_mcqa_predictions_qwen14_top20.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
model.eval()


done = set()
if OUTPUT.exists():
    with open(OUTPUT, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"(dev_mcqa_case_\d+)", line)
            if m:
                done.add(m.group(1))


with open(INPUT, encoding="utf-8") as f, \
     open(OUTPUT, "a", encoding="utf-8") as out:

    for line in f:
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

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

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

        m_ans = re.search(r"Answer:\s*([A-Z])", response, re.I)
        if m_ans:
             pred = m_ans.group(1).upper() 
        else:
             m_ans = re.search(r"\b([A-Z])\b", response)
             pred = m_ans.group(1).upper() if m_ans else "NONE"

        out.write(f"{cid}\tAnswer: {pred}\n")
        out.flush()

        print(cid, pred)
