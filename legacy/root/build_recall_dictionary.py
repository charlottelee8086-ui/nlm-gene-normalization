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
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

gold_mentions = []
for doc in read_arrow(gold_path):
    pmid = doc["document_id"]
    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue
        if not ent.get("normalized"):
            continue
        start, end = ent["offsets"][0]
        mention = ent["text"][0]
        gold_mentions.append((pmid, start, end, mention))

pred_spans = set()
with open(pred_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        pmid, start, end, mention, etype, gid = parts[:6]
        if etype == "Gene":
            pred_spans.add((pmid, int(start), int(end)))

missed = []
for pmid, start, end, mention in gold_mentions:
    if (pmid, start, end) not in pred_spans:
        missed.append(mention)

counter = Counter(missed)

with open("recall_dictionary_candidates.tsv", "w", encoding="utf-8") as out:
    out.write("mention\tcount\n")
    for mention, count in counter.most_common():
        if count >= 3:
            out.write(f"{mention}\t{count}\n")

print("Saved recall_dictionary_candidates.tsv")
print("Top candidates:")
for m, c in counter.most_common(30):
    print(c, m)
