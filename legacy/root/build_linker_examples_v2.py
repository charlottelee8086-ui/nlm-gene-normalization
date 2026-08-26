import json
import random
import pyarrow as pa
import pyarrow.ipc as ipc
from collections import defaultdict, Counter

random.seed(13)

TRAIN = "nlm_gene-train.arrow"

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

mention2ids = defaultdict(Counter)
gid2names = defaultdict(Counter)
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
                gid2names[gid][mention] += 1
                all_gene_ids[gid] += 1
                ents.append((start, end, mention, gid))

    docs.append({"pmid": doc["document_id"], "text": text, "entities": ents})

gid2bestname = {
    gid: names.most_common(1)[0][0]
    for gid, names in gid2names.items()
}

all_ids = list(all_gene_ids.keys())

examples = []

for doc in docs:
    text = doc["text"]

    for start, end, mention, gold_gid in doc["entities"]:
        key = mention.lower()
        context = get_context(text, start, end)

        candidates = list(mention2ids[key].keys())
        if gold_gid not in candidates:
            candidates.append(gold_gid)

        negatives = [gid for gid in candidates if gid != gold_gid]

        while len(negatives) < 3:
            gid = random.choice(all_ids)
            if gid != gold_gid and gid not in negatives:
                negatives.append(gid)

        examples.append({
            "context": context,
            "mention": mention,
            "candidate_gene_id": gold_gid,
            "candidate_name": gid2bestname[gold_gid],
            "label": 1,
        })

        for gid in negatives[:3]:
            examples.append({
                "context": context,
                "mention": mention,
                "candidate_gene_id": gid,
                "candidate_name": gid2bestname.get(gid, "unknown gene"),
                "label": 0,
            })

random.shuffle(examples)

split = int(len(examples) * 0.9)

for path, data in [
    ("linker_train_v2.jsonl", examples[:split]),
    ("linker_dev_v2.jsonl", examples[split:]),
]:
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("train:", split)
print("dev:", len(examples) - split)
print("gene ids:", len(gid2bestname))
