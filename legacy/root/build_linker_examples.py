import json
import random
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from collections import defaultdict, Counter

random.seed(13)

TRAIN = "nlm_gene-train.arrow"
OUT_TRAIN = "linker_train.jsonl"
OUT_DEV = "linker_dev.jsonl"

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
    return text[left:start] + " [MENTION] " + text[start:end] + " [/MENTION] " + text[end:right]

rows = read_arrow(TRAIN)

# mention string -> candidate gene ids from train
mention2ids = defaultdict(Counter)
all_gene_ids = Counter()

docs = []

for doc in rows:
    text = make_text(doc)
    ents = []

    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        if not ent.get("normalized"):
            continue

        mention = ent["text"][0].strip()
        start, end = ent["offsets"][0]

        for norm in ent["normalized"]:
            if norm.get("db_name") == "NCBIGene":
                gid = norm["db_id"]
                mention2ids[mention.lower()][gid] += 1
                all_gene_ids[gid] += 1
                ents.append((start, end, mention, gid))

    docs.append({
        "pmid": doc["document_id"],
        "text": text,
        "entities": ents,
    })

all_ids = list(all_gene_ids.keys())

examples = []

for doc in docs:
    text = doc["text"]

    for start, end, mention, gold_gid in doc["entities"]:
        key = mention.lower()

        # candidate ids: same surface form candidates from train
        candidates = list(mention2ids[key].keys())

        # guarantee positive candidate
        if gold_gid not in candidates:
            candidates.append(gold_gid)

        # hard negatives: other IDs seen with same mention
        negatives = [gid for gid in candidates if gid != gold_gid]

        # random negatives
        while len(negatives) < 5:
            gid = random.choice(all_ids)
            if gid != gold_gid and gid not in negatives:
                negatives.append(gid)

        context = get_context(text, start, end)

        # positive
        examples.append({
            "text": context,
            "mention": mention,
            "candidate_gene_id": gold_gid,
            "label": 1,
        })

        # negatives
        for gid in negatives[:1]:
            examples.append({
                "text": context,
                "mention": mention,
                "candidate_gene_id": gid,
                "label": 0,
            })

random.shuffle(examples)

split = int(len(examples) * 0.9)
train_ex = examples[:split]
dev_ex = examples[split:]

with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for x in train_ex:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open(OUT_DEV, "w", encoding="utf-8") as f:
    for x in dev_ex:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("train examples:", len(train_ex))
print("dev examples:", len(dev_ex))
print("unique gene ids:", len(all_ids))
print("unique mention strings:", len(mention2ids))
