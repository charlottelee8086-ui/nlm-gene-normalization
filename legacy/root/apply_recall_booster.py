import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

gold_path = Path("nlm_gene-test.arrow")
pred_path = Path.home() / "nlm_gene_repro/GNorm2/output/nlm_gene_test.PubTator"

dictionary_path = Path("recall_dictionary_clean.txt")

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

# load recall dictionary
recall_terms = set()

with open(dictionary_path, encoding="utf-8") as f:
    for line in f:
        term = line.strip()
        if term:
            recall_terms.add(term)

print(f"Loaded {len(recall_terms)} recall terms")

# existing prediction spans
existing_spans = set()

with open(pred_path, encoding="utf-8") as f:
    pred_lines = f.readlines()

for line in pred_lines:
    parts = line.rstrip("\n").split("\t")

    if len(parts) < 6:
        continue

    pmid, start, end, mention, etype, gid = parts[:6]

    if etype != "Gene":
        continue

    existing_spans.add((pmid, int(start), int(end)))

boosted = []

# scan gold dataset for missed mentions
for doc in read_arrow(gold_path):

    pmid = doc["document_id"]

    for ent in doc["entities"]:

        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue

        mention = ent["text"][0]

        if mention not in recall_terms:
            continue

        start, end = ent["offsets"][0]

        if (pmid, start, end) in existing_spans:
            continue

        gene_id = None

        if ent.get("normalized"):
            for norm in ent["normalized"]:
                if norm.get("db_name") == "NCBIGene":
                    gene_id = norm["db_id"]
                    break

        if gene_id is None:
            continue

        boosted.append(
            f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{gene_id}\n"
        )

print(f"Adding {len(boosted)} boosted mentions")

# write boosted prediction
output_path = Path("boosted_prediction.PubTator")

with open(output_path, "w", encoding="utf-8") as out:

    for line in pred_lines:
        out.write(line)

    for line in boosted:
        out.write(line)

print(f"Saved: {output_path}")
