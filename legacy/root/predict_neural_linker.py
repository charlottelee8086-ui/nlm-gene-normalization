import json
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from collections import defaultdict, Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_DIR = "pubmedbert_ncbigene_linker_v2_best"

NER_PRED = "pubmedbert_ner_test_predictions.tsv"

TEST_ARROW = "nlm_gene-test.arrow"

OUT = "pubmedbert_neural_linked.PubTator"


def read_arrow(path):
    with pa.memory_map(path, "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def make_text(doc):
    passages = sorted(doc["passages"], key=lambda p: p["offsets"][0][0])
    text = ""
    for p in passages:
        start = p["offsets"][0][0]
        content = p["text"][0]
        if len(text) < start:
            text += " " * (start - len(text))
        text += content
    return text


def get_context(text, start, end, window=160):
    left = max(0, start - window)
    right = min(len(text), end + window)

    return (
        text[left:start]
        + " [MENTION] "
        + text[start:end]
        + " [/MENTION] "
        + text[end:right]
    )


# build candidate KB from train
train_rows = read_arrow("nlm_gene-train.arrow")

mention2ids = defaultdict(Counter)
gid2names = defaultdict(Counter)

for doc in train_rows:

    for ent in doc["entities"]:

        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue

        if not ent.get("normalized"):
            continue

        mention = ent["text"][0].strip()

        for norm in ent["normalized"]:

            if norm.get("db_name") == "NCBIGene":

                gid = norm["db_id"]

                mention2ids[mention.lower()][gid] += 1
                gid2names[gid][mention] += 1

gid2bestname = {
    gid: names.most_common(1)[0][0]
    for gid, names in gid2names.items()
}

# load test texts
test_rows = read_arrow(TEST_ARROW)

pmid2text = {}

for doc in test_rows:
    pmid2text[doc["document_id"]] = make_text(doc)

# load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

# run linker
out = open(OUT, "w", encoding="utf-8")

linked = 0
unlinked = 0

with open(NER_PRED, encoding="utf-8") as f:

    for line in f:

        parts = line.rstrip().split("\t")

        if len(parts) < 5:
            continue

        pmid, start, end, mention, etype = parts[:5]

        if etype != "Gene":
            continue

        start = int(start)
        end = int(end)

        candidates = list(
            mention2ids[mention.lower()].keys()
        )

        if not candidates:
            unlinked += 1
            continue

        context = get_context(
            pmid2text[pmid],
            start,
            end
        )

        best_gid = None
        best_score = -999

        for gid in candidates:

            candidate_name = gid2bestname.get(
                gid,
                "unknown gene"
            )

            text = (
                context
                + " [SEP] Mention: "
                + mention
                + " [SEP] Candidate: "
                + candidate_name
                + " [SEP] Candidate Gene ID: "
                + str(gid)
            )

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                logits = model(**inputs).logits

            score = logits[0][1].item()

            if score > best_score:
                best_score = score
                best_gid = gid

        out.write(
            f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{best_gid}\n"
        )

        linked += 1

print("linked:", linked)
print("unlinked:", unlinked)

out.close()

print("saved:", OUT)
