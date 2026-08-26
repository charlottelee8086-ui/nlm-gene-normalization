import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

GOLD_ARROW = Path("nlm_gene-test.arrow")
PRED = Path.home() / "nlm_gene_repro/GNorm2/gold_norm_output/nlm_gene_test.PubTator"
RESCUE = Path("rescue_dictionary.tsv")
OUT = Path("still_missing_after_rescue.tsv")


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def clean_gid(x):
    x = x.replace("NCBIGene:", "").replace("*", "").strip()
    if "|" in x:
        x = x.split("|")[0]
    if "," in x:
        x = x.split(",")[0]
    return x


def load_gold():
    rows = []
    for doc in read_arrow(GOLD_ARROW):
        pmid = str(doc["document_id"])
        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            mention = ent["text"][0].strip()
            start, end = ent["offsets"][0]

            gids = []
            for norm in ent.get("normalized", []):
                if norm.get("db_name") == "NCBIGene":
                    gids.append(str(norm["db_id"]))

            if gids:
                rows.append((pmid, int(start), int(end), mention, "|".join(sorted(set(gids)))))

    return rows


def load_pred_spans():
    pred = set()
    with open(PRED, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            pmid, start, end, mention, etype, gid = parts[:6]
            if etype != "Gene":
                continue
            gid = clean_gid(gid)
            if gid and gid[0].isdigit():
                pred.add((pmid, int(start), int(end)))
    return pred


def load_rescue():
    rescue = {}
    with open(RESCUE, encoding="utf-8") as f:
        next(f)
        for line in f:
            mention, gid, freq, ratio = line.rstrip("\n").split("\t")
            rescue[mention] = gid
    return rescue


gold_rows = load_gold()
pred_spans = load_pred_spans()
rescue = load_rescue()

still = []

for pmid, start, end, mention, gids in gold_rows:
    key = (pmid, start, end)

    if key in pred_spans:
        continue

    if mention in rescue:
        continue

    still.append((pmid, start, end, mention, gids))

with open(OUT, "w", encoding="utf-8") as out:
    out.write("pmid\tstart\tend\tmention\tgold_gene_ids\n")
    for row in still:
        out.write("{}\t{}\t{}\t{}\t{}\n".format(*row))

print("saved:", OUT)
print("still missing:", len(still))
