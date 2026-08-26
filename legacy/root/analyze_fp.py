import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from collections import Counter

gold_path = Path("nlm_gene-test.arrow")
pred_path = Path.home() / "nlm_gene_repro/GNorm2/output/nlm_gene_test.PubTator"

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

gold = set()

for doc in read_arrow(gold_path):
    pmid = doc["document_id"]
    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        for norm in ent.get("normalized", []):
            if norm.get("db_name") == "NCBIGene":
                start, end = ent["offsets"][0]
                gold.add((pmid, start, end, norm["db_id"]))

pred = set()
pred_full = []

with open(pred_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split("\t")
        if len(parts) < 6:
            continue
        pmid, start, end, mention, etype, gid = parts[:6]
        if etype != "Gene":
            continue
        item = (pmid, int(start), int(end), gid)
        pred.add(item)
        pred_full.append((pmid, int(start), int(end), mention, gid))

fp = pred - gold

counter = Counter()

for pmid, start, end, mention, gid in pred_full:
    if (pmid, start, end, gid) in fp:
        counter[mention] += 1

print("Top False Positive Mentions\n")

for mention, count in counter.most_common(50):
    print(count, mention)
