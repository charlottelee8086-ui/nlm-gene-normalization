import pyarrow as pa
import pyarrow.ipc as ipc
from collections import Counter, defaultdict

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

mention2ids = defaultdict(list)

for doc in rows:

    for ent in doc["entities"]:

        if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
            continue

        if not ent.get("normalized"):
            continue

        mention = ent["text"][0].lower()

        for norm in ent["normalized"]:

            if norm["db_name"] != "NCBIGene":
                continue

            gene_id = norm["db_id"]

            mention2ids[mention].append(gene_id)

with open("linking_dictionary.tsv", "w", encoding="utf-8") as out:

    for mention, ids in mention2ids.items():

        best_id = Counter(ids).most_common(1)[0][0]

        out.write(f"{mention}\t{best_id}\n")

print("saved linking_dictionary.tsv")
print("dictionary size:", len(mention2ids))
