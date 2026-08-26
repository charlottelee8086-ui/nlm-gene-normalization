import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
import json

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
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

def collect_entities(doc):
    ents = []
    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        if not ent.get("normalized"):
            continue
        start, end = ent["offsets"][0]
        ents.append((start, end))
    return ents

def convert(split):
    rows = read_arrow(f"nlm_gene-{split}.arrow")
    out = []

    for doc in rows:
        text = make_text(doc)
        ents = collect_entities(doc)
        out.append({
            "id": doc["document_id"],
            "text": text,
            "entities": ents
        })

    Path(f"ner_{split}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out),
        encoding="utf-8"
    )

convert("train")
convert("test")

print("saved ner_train.jsonl and ner_test.jsonl")
