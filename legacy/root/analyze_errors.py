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
rows = read_arrow(gold_path)

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
            if norm["db_name"] == "NCBIGene":
                gold.add((pmid, start, end, mention, norm["db_id"]))

pred = set()

with open(pred_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split("\t")

        if len(parts) < 6:
            continue

        pmid, start, end, mention, etype, gid = parts[:6]

        if etype != "Gene":
            continue

        pred.add((pmid, int(start), int(end), mention, gid))

gold_norm = {(p,s,e,gid) for p,s,e,m,gid in gold}
pred_norm = {(p,s,e,gid) for p,s,e,m,gid in pred}

fn = gold_norm - pred_norm

mention_counter = Counter()

for p,s,e,gid in fn:
    for x in gold:
        if x[0]==p and x[1]==s and x[2]==e and x[4]==gid:
            mention_counter[x[3]] += 1

print("Top False Negative Mentions")
print()

for m,c in mention_counter.most_common(50):
    print(c, m)
