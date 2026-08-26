from pathlib import Path
import random
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

ARROW = Path("nlm_gene-train.arrow")
OUT_TRAIN = Path("bioelqa_train_mentions.tsv")
OUT_DEV = Path("bioelqa_dev_mentions.tsv")

random.seed(42)

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

def get_doc_text(doc):
    texts = []
    for p in doc["passages"]:
        # BigBio passages text is often a list
        txt = p["text"]
        if isinstance(txt, list):
            txt = txt[0]
        texts.append(txt)
    return " ".join(texts)

def window_context(text, start, end, window=500):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].replace("\n", " ")

rows = read_arrow(ARROW)
records = []

for doc in rows:
    doc_id = doc.get("document_id", doc.get("id", ""))
    full_text = get_doc_text(doc)

    for ent in doc.get("entities", []):
        mention = ent.get("text")
        if isinstance(mention, list):
            mention = mention[0]

        offsets = ent.get("offsets", [])
        if not offsets:
            continue

        start, end = offsets[0]
        norm_ids = ent.get("normalized", [])

        # BigBio often stores normalized IDs as list of dicts
        gene_ids = []
        for n in norm_ids:
            db_id = n.get("db_id") or n.get("id")
            if db_id:
                gene_ids.append(str(db_id).replace("NCBI Gene:", "").replace("GeneID:", ""))

        if not gene_ids:
            continue

        records.append({
            "doc_id": doc_id,
            "mention": mention,
            "start": int(start),
            "end": int(end),
            "gold_geneid": "|".join(sorted(set(gene_ids))),
            "context": window_context(full_text, int(start), int(end), 500),
        })

df = pd.DataFrame(records)

# document-level split，避免同一篇文章的 mentions 同时进 train/dev
doc_ids = sorted(df["doc_id"].unique())
random.shuffle(doc_ids)

n_dev = max(1, int(len(doc_ids) * 0.2))
dev_docs = set(doc_ids[:n_dev])

dev_df = df[df["doc_id"].isin(dev_docs)].reset_index(drop=True)
train_df = df[~df["doc_id"].isin(dev_docs)].reset_index(drop=True)

train_df.to_csv(OUT_TRAIN, sep="\t", index=False)
dev_df.to_csv(OUT_DEV, sep="\t", index=False)

print("Total mentions:", len(df))
print("Train mentions:", len(train_df))
print("Dev mentions:", len(dev_df))
print("Train docs:", train_df["doc_id"].nunique())
print("Dev docs:", dev_df["doc_id"].nunique())
print("Saved:", OUT_TRAIN, OUT_DEV)
print("\nSample:")
print(dev_df.head(3).to_string())
