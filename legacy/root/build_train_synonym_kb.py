import pyarrow as pa
import pyarrow.ipc as ipc

def read_arrow(path):
    with pa.memory_map(path, "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()

rows = read_arrow("nlm_gene-train.arrow")

pairs = set()

for doc in rows:

    for ent in doc["entities"]:

        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue

        if not ent.get("normalized"):
            continue

        mention = ent["text"][0].strip()

        for norm in ent["normalized"]:

            if norm.get("db_name") == "NCBIGene":

                gid = norm["db_id"]

                pairs.add((mention, gid))

with open("train_synonym_kb.tsv", "w", encoding="utf-8") as out:

    for mention, gid in sorted(pairs):

        out.write(f"{mention}\t{gid}\n")

print("KB synonyms:", len(pairs))
