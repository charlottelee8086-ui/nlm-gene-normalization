import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

gold_path = Path("nlm_gene-test.arrow")
pred_path = Path("pubmedbert_neural_linked_v2.PubTator")

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

def load_gold():
    rows = read_arrow(gold_path)
    gold = set()

    for doc in rows:
        pmid = doc["document_id"]
        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            if not ent.get("normalized"):
                continue

            start, end = ent["offsets"][0]
            mention = ent["text"][0]

            for norm in ent["normalized"]:
                if norm.get("db_name") == "NCBIGene":
                    gene_id = norm["db_id"]
                    gold.add((pmid, start, end, mention, gene_id))

    return gold

def load_pred():
    pred = set()

    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            pmid, start, end, mention, etype, gene_id = parts[:6]

            if etype != "Gene":
                continue

            pred.add((pmid, int(start), int(end), mention, gene_id))

    return pred

gold = load_gold()
pred = load_pred()

gold_norm = {(p, s, e, gid) for p, s, e, m, gid in gold}
pred_norm = {(p, s, e, gid) for p, s, e, m, gid in pred}

tp = len(gold_norm & pred_norm)
fp = len(pred_norm - gold_norm)
fn = len(gold_norm - pred_norm)

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("Gold:", len(gold_norm))
print("Pred:", len(pred_norm))
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
