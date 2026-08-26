import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

GOLD_ARROW = Path("nlm_gene-test.arrow")
PRED = Path.home() / "nlm_gene_repro/GNorm2/gold_norm_output/nlm_gene_test.PubTator"


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
    gold = {}

    for doc in read_arrow(GOLD_ARROW):
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


def clean_gid(x):
    x = x.replace("NCBIGene:", "")
    x = x.replace("*", "")
    x = x.strip()

    # sometimes output may contain multiple ids
    if "|" in x:
        x = x.split("|")[0]
    if "," in x:
        x = x.split(",")[0]

    return x


def load_pred():
    pred = {}

    with open(PRED, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 6:
                continue

            pmid, start, end, mention, etype, gid = parts[:6]

            if etype != "Gene":
                continue

            gid = clean_gid(gid)

            if not gid or not gid[0].isdigit():
                continue

            pred[(pmid, int(start), int(end))] = gid

    return pred


gold = load_gold()
pred = load_pred()

correct = 0
wrong = 0
missing = 0

for span, gold_gids in gold.items():
    if span not in pred:
        missing += 1
        continue

    if pred[span] in gold_gids:
        correct += 1
    else:
        wrong += 1

total = len(gold)

print("Gold mentions:", total)
print("Pred normalized:", len(pred))
print("Correct:", correct)
print("Wrong:", wrong)
print("Missing normalization:", missing)

print("Accuracy on all gold mentions:", correct / total)
print("Accuracy on normalized mentions only:", correct / (correct + wrong) if correct + wrong else 0)
