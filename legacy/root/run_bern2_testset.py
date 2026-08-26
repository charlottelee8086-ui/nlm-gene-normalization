import json
import requests
import time

INPUT = "ner_test.jsonl"
OUTPUT = "bern2_test_predictions.PubTator"

url = "http://bern2.korea.ac.kr/plain"

docs = []

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print("docs:", len(docs))

out = open(OUTPUT, "w", encoding="utf-8")

for i, doc in enumerate(docs):

    pmid = doc["id"]
    text = doc["text"]

    try:

        r = requests.post(
            url,
            json={"text": text},
            timeout=120
        )

        data = r.json()

    except Exception as e:

        print("ERROR", pmid, e)
        continue

    anns = data.get("annotations", [])

    for ann in anns:

        if ann.get("obj") != "gene":
            continue

        ids = ann.get("id", [])

        if not ids:
            continue

        gene_id = ids[0]

        if not gene_id.startswith("NCBIGene:"):
            continue

        gene_id = gene_id.replace("NCBIGene:", "")

        mention = ann["mention"]

        start = ann["span"]["begin"]
        end = ann["span"]["end"]

        out.write(
            f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{gene_id}\n"
        )

    if (i + 1) % 10 == 0:
        print(i + 1)

    time.sleep(0.5)

out.close()

print("saved:", OUTPUT)
