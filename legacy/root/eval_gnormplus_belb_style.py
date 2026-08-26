import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

gold_path = Path("nlm_gene-test.arrow")
pred_path = Path("/mnt/beegfs/home/xli/nlm_gene_repro/GNorm2/output/nlm_gene_test.PubTator")

def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def load_gold():
    rows = read_arrow(gold_path)
    gold = {}

    for doc in rows:
        pmid = str(doc["document_id"])

        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            if not ent.get("normalized"):
                continue

            start, end = ent["offsets"][0]

            gids = set()
            for norm in ent["normalized"]:
                if norm.get("db_name") == "NCBIGene":
                    gids.add(str(norm["db_id"]))

            if gids:
                gold[(pmid, int(start), int(end))] = gids

    return gold


def load_pred():
    pred = {}

    with open(pred_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 6:
                continue

            pmid, start, end, mention, etype, gid = parts[:6]

            if etype != "Gene":
                continue

            gid = gid.replace("NCBIGene:", "").replace("*", "").strip()

            if not gid or not gid[0].isdigit():
                continue

            pred[(pmid, int(start), int(end))] = gid

    return pred


gold = load_gold()
pred = load_pred()

recognized = 0
correct_link = 0
wrong_link = 0

for span, gold_gids in gold.items():
    if span in pred:
        recognized += 1
        if pred[span] in gold_gids:
            correct_link += 1
        else:
            wrong_link += 1

print("Gold mentions:", len(gold))
print("Predicted gene mentions:", len(pred))
print("Correctly recognized gold spans:", recognized)
print("Correct links among recognized spans:", correct_link)
print("Wrong links among recognized spans:", wrong_link)

if recognized:
    print("BELB-style linking recall@1 / accuracy:", correct_link / recognized)
