from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

ARROW = Path("nlm_gene-train.arrow")
TRAIN_TSV = Path("bioelqa_train_mentions.tsv")
DEV_TSV = Path("bioelqa_dev_mentions.tsv")


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def get_text(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)


def get_abstract_text(doc):
    parts = []
    for p in doc.get("passages", []):
        if str(p.get("type", "")).lower() == "abstract":
            parts.append(get_text(p.get("text", "")))
    return " ".join(parts).strip()


def summarize(df, name):
    print("=" * 80)
    print(name)
    print("Documents:", len(df))
    print("Average abstract length, words:", round(df["abstract_words"].mean(), 2))
    print("Median abstract length, words:", round(df["abstract_words"].median(), 2))
    print("Min abstract length, words:", int(df["abstract_words"].min()))
    print("Max abstract length, words:", int(df["abstract_words"].max()))
    print("Average abstract length, chars:", round(df["abstract_chars"].mean(), 2))


rows = read_arrow(ARROW)

records = []

for doc in rows:
    doc_id = str(doc.get("document_id", doc.get("id", "")))
    abstract = get_abstract_text(doc)

    records.append({
        "doc_id": doc_id,
        "abstract_text": abstract,
        "abstract_words": len(abstract.split()),
        "abstract_chars": len(abstract),
    })

doc_df = pd.DataFrame(records)

summarize(doc_df, "All documents")

if TRAIN_TSV.exists() and DEV_TSV.exists():
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    dev_df = pd.read_csv(DEV_TSV, sep="\t")

    train_docs = set(train_df["doc_id"].astype(str))
    dev_docs = set(dev_df["doc_id"].astype(str))

    train_doc_df = doc_df[doc_df["doc_id"].astype(str).isin(train_docs)]
    dev_doc_df = doc_df[doc_df["doc_id"].astype(str).isin(dev_docs)]

    summarize(train_doc_df, "Train documents")
    summarize(dev_doc_df, "Dev documents")

    print("=" * 80)
    print("Overlapping docs:", len(train_docs & dev_docs))

doc_df.to_csv("nlm_gene_abstract_lengths.tsv", sep="\t", index=False)
print("Saved: nlm_gene_abstract_lengths.tsv")
