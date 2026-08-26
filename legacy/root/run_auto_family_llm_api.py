import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


INPUT = Path("auto_family_llm_prompts.jsonl")
OUTPUT = Path("auto_family_llm_predictions_api.txt")
RAW_OUTPUT = Path("auto_family_llm_responses_api.jsonl")

MODEL = "gpt-4o-mini"


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def parse_gene_id(text):
    m = re.search(r"GeneID:\s*(\d+)", text)
    if m:
        return m.group(1)
    return ""


done = set()

if OUTPUT.exists():
    with open(OUTPUT, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"(auto_family_case_\d+)", line)
            if m:
                done.add(m.group(1))


with open(INPUT, encoding="utf-8") as f, \
     open(OUTPUT, "a", encoding="utf-8") as out_pred, \
     open(RAW_OUTPUT, "a", encoding="utf-8") as out_raw:

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
                    "You must choose exactly one Gene ID from the provided candidate list. "
                    "Do not invent Gene IDs. "
                    "Answer only in the requested format."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                )

                text = resp.choices[0].message.content.strip()
                gid = parse_gene_id(text)

                if not gid:
                    text = text.replace("\n", " ")
                    gid = parse_gene_id(text)

                out_pred.write(
                    "{}    GeneID: {}\n".format(case_id, gid)
                )
                out_pred.flush()

                out_raw.write(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "response": text,
                            "parsed_gene_id": gid,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out_raw.flush()

                print(case_id, gid)
                break

            except Exception as e:
                print("error", case_id, attempt, e)
                time.sleep(5)

        time.sleep(0.5)

print("saved:", OUTPUT)
print("saved:", RAW_OUTPUT)
