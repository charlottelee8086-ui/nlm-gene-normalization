import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from collections import defaultdict, Counter

OUT = Path("gene_id_name_map.tsv")


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


gid2names = defaultdict(Counter)

for arrow_file in ["nlm_gene-train.arrow", "nlm_gene-test.arrow"]:
    for doc in read_arrow(arrow_file):
        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            mention = ent["text"][0].strip()

            for norm in ent.get("normalized", []):
                if norm.get("db_name") == "NCBIGene":
                    gid = str(norm["db_id"])
                    gid2names[gid][mention] += 1

with open(OUT, "w", encoding="utf-8") as out:
    out.write("gene_id\tname\n")
    for gid, counter in gid2names.items():
        name = counter.most_common(1)[0][0]
        out.write(f"{gid}\t{name}\n")

print("saved:", OUT)
print("gene ids:", len(gid2names))

