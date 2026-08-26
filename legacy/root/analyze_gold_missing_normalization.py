from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

GOLD_ARROW = "nlm_gene-test.arrow"
PRED_FILE = str(
    Path.home() /
    "nlm_gene_repro/GNorm2/gold_norm_output/nlm_gene_test.PubTator"
)


def read_arrow(path):
    with pa.memory_map(path, "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
        return reader.read_all().to_pylist()


# -------------------------
# Gold mentions
# -------------------------

gold = {}

for doc in read_arrow(GOLD_ARROW):
    pmid = str(doc["document_id"])

    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue

        mention = ent["text"][0].strip()
        start, end = ent["offsets"][0]

        gold[(pmid, int(start), int(end))] = mention


# -------------------------
# Predicted normalized spans
# -------------------------

pred = set()

with open(PRED_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.rstrip().split("\t")

        if len(parts) < 6:
            continue

        pmid, start, end, mention, etype, gid = parts[:6]

        if etype != "Gene":
            continue

        gid = gid.replace("NCBIGene:", "").strip()

        if gid and gid[0].isdigit():
            pred.add((pmid, int(start), int(end)))


# -------------------------
# Missing normalization
# -------------------------

missing_mentions = []

for span, mention in gold.items():
    if span not in pred:
        missing_mentions.append(mention)

print("Missing normalization:", len(missing_mentions))

counter = Counter(missing_mentions)

print("\n=== Top missing mentions ===\n")

for mention, freq in counter.most_common(100):
    print(freq, mention)
