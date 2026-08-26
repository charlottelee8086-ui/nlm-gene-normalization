import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

gold_path = Path("nlm_gene-test.arrow")
pred_path = Path("pubmedbert_ner_test_predictions.tsv")

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

gold = set()
for doc in read_arrow(gold_path):
    pmid = doc["document_id"]
    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        if not ent.get("normalized"):
            continue
        start, end = ent["offsets"][0]
        gold.add((pmid, start, end))

pred = set()
with open(pred_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        pmid, start, end, mention, etype = parts[:5]
        if etype != "Gene":
            continue
        pred.add((pmid, int(start), int(end)))

tp = len(gold & pred)
fp = len(pred - gold)
fn = len(gold - pred)

p = tp / (tp + fp) if tp + fp else 0
r = tp / (tp + fn) if tp + fn else 0
f1 = 2*p*r/(p+r) if p+r else 0

print("Gold spans:", len(gold))
print("Pred spans:", len(pred))
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1:        {f1:.4f}")
