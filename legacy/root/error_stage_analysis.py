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

        for norm in ent["normalized"]:
            if norm.get("db_name") == "NCBIGene":
                gold.append({
                    "pmid": pmid,
                    "start": start,
                    "end": end,
                    "mention": mention,
                    "gene_id": norm["db_id"],
                    "type": ent["type"],
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
            "mention": mention,
            "gene_id": gid,
        })

# For NER-level matching, ignore Gene ID
pred_spans = {(p["pmid"], p["start"], p["end"]) for p in pred}

# For exact end-to-end matching
pred_full = {(p["pmid"], p["start"], p["end"], p["gene_id"]) for p in pred}

gold_spans = {(g["pmid"], g["start"], g["end"]) for g in gold}
gold_full = {(g["pmid"], g["start"], g["end"], g["gene_id"]) for g in gold}

correct = gold_full & pred_full

ner_miss = []
linking_error = []

for g in gold:
    span_key = (g["pmid"], g["start"], g["end"])
    full_key = (g["pmid"], g["start"], g["end"], g["gene_id"])

    if full_key in pred_full:
        continue

    if span_key not in pred_spans:
        ner_miss.append(g)
    else:
        # span detected, but gold Gene ID not predicted
        linking_error.append(g)

print("=== Error Stage Decomposition ===")
print(f"Gold entities: {len(gold_full)}")
print(f"Pred entities: {len(pred_full)}")
print(f"Correct exact span+ID: {len(correct)}")
print(f"NER misses: {len(ner_miss)}")
print(f"Linking errors: {len(linking_error)}")
print()

denom = len(gold_full)
print("=== Percent of Gold Entities ===")
print(f"Correct:        {len(correct)/denom:.3f}")
print(f"NER miss:       {len(ner_miss)/denom:.3f}")
print(f"Linking error:  {len(linking_error)/denom:.3f}")
print()

error_denom = len(ner_miss) + len(linking_error)
print("=== Percent of Errors Only ===")
print(f"NER miss:       {len(ner_miss)/error_denom:.3f}")
print(f"Linking error:  {len(linking_error)/error_denom:.3f}")
print()

print("=== Top NER Miss Mentions ===")
for mention, count in Counter(g["mention"] for g in ner_miss).most_common(30):
    print(count, mention)

print()
print("=== Top Linking Error Mentions ===")
for mention, count in Counter(g["mention"] for g in linking_error).most_common(30):
    print(count, mention)
