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

gold = []
for doc in read_arrow(gold_path):
    pmid = doc["document_id"]

    for ent in doc["entities"]:
        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue

        if not ent.get("normalized"):
            continue

        start, end = ent["offsets"][0]
        mention = ent["text"][0]

        gold.append({
            "pmid": pmid,
            "start": start,
            "end": end,
            "mention": mention
        })

pred = []
with open(pred_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 6:
            continue

        pmid, start, end, mention, etype, gid = parts[:6]

        if etype != "Gene":
            continue

        pred.append({
            "pmid": pmid,
            "start": int(start),
            "end": int(end),
            "mention": mention
        })

pred_spans = {(p["pmid"], p["start"], p["end"]) for p in pred}

exact_match = 0
partial_overlap = []
true_miss = []

for g in gold:

    span_key = (g["pmid"], g["start"], g["end"])

    if span_key in pred_spans:
        exact_match += 1
        continue

    found_overlap = False

    for p in pred:

        if g["pmid"] != p["pmid"]:
            continue

        overlap = not (
            g["end"] <= p["start"] or
            p["end"] <= g["start"]
        )

        if overlap:
            found_overlap = True
            partial_overlap.append((g, p))
            break

    if not found_overlap:
        true_miss.append(g)

print("=== Span Analysis ===")
print(f"Gold mentions: {len(gold)}")
print(f"Exact span match: {exact_match}")
print(f"Partial overlap: {len(partial_overlap)}")
print(f"True miss: {len(true_miss)}")
print()

total_errors = len(partial_overlap) + len(true_miss)

print("=== Error Breakdown ===")
print(f"Partial overlap: {len(partial_overlap)/total_errors:.3f}")
print(f"True miss: {len(true_miss)/total_errors:.3f}")
print()

print("=== Top Partial Overlap Mentions ===")
for mention, count in Counter(g['mention'] for g, p in partial_overlap).most_common(30):
    print(count, mention)

print()
print("=== Top True Miss Mentions ===")
for mention, count in Counter(g['mention'] for g in true_miss).most_common(30):
    print(count, mention)
