import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "pubmedbert_nlm_gene_ner_best"
TEST_FILE = "ner_test.jsonl"
OUT_FILE = "pubmedbert_ner_test_predictions.tsv"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

id2label = model.config.id2label

with open(TEST_FILE, encoding="utf-8") as f, open(OUT_FILE, "w", encoding="utf-8") as out:

    for line in f:
        doc = json.loads(line)

        pmid = doc["id"]
        text = doc["text"]

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        offsets = encoding.pop("offset_mapping")[0]

        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        preds = outputs.logits.argmax(-1)[0].cpu().numpy()

        current = None

        for pred, (start, end) in zip(preds, offsets):

            label = id2label[int(pred)]

            if start == end:
                continue

            if label == "B-GENE":

                if current:
                    s, e = current
                    mention = text[s:e]
                    out.write(f"{pmid}\t{s}\t{e}\t{mention}\tGene\n")

                current = [int(start), int(end)]

            elif label == "I-GENE" and current:
                current[1] = int(end)

            else:
                if current:
                    s, e = current
                    mention = text[s:e]
                    out.write(f"{pmid}\t{s}\t{e}\t{mention}\tGene\n")
                    current = None

        if current:
            s, e = current
            mention = text[s:e]
            out.write(f"{pmid}\t{s}\t{e}\t{mention}\tGene\n")

print("Saved:", OUT_FILE)
