import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL = "Qwen/Qwen2.5-14B-Instruct"
INPUT = Path("auto_family_llm_prompts.jsonl")
OUTPUT = Path("auto_family_llm_predictions_qwen14_nocurrent.txt")
RAW = Path("auto_family_llm_responses_qwen14_nocurrent.jsonl")


def parse_gene_id(text):
    m = re.search(r"GeneID:\s*(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{2,12})\b", text)
    if m:
        return m.group(1)
    return ""


print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

done = set()

if OUTPUT.exists():
    with open(OUTPUT, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"(auto_family_case_\d+)", line)
            if m:
                done.add(m.group(1))

with open(INPUT, encoding="utf-8") as f, \
     open(OUTPUT, "a", encoding="utf-8") as out, \
     open(RAW, "a", encoding="utf-8") as raw:

    for line in f:
        ex = json.loads(line)

        case_id = ex["case_id"]

        if case_id in done:
            continue

        prompt = ex["prompt"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical gene normalization assistant. "
                    "Choose exactly one Gene ID from the provided candidates. "
                    "Do not invent Gene IDs. "
                    "Answer only in the requested format."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        gid = parse_gene_id(response)

        out.write(f"{case_id}    GeneID: {gid}\n")
        out.flush()

        raw.write(
            json.dumps(
                {
                    "case_id": case_id,
                    "response": response,
                    "parsed_gene_id": gid,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        raw.flush()

        print(case_id, gid)

print("saved:", OUTPUT)
print("saved:", RAW)
