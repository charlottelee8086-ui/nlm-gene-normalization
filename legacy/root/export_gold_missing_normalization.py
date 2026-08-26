from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

GOLD_ARROW = Path("nlm_gene-test.arrow")
PRED_FILE = Path.home() / "nlm_gene_repro/GNorm2/gold_norm_output/nlm_gene_test.PubTator"
OUT = Path("gold_missing_normalization.tsv")


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

            rows.append({
                "pmid": pmid,
                "start": int(start),
                "end": int(end),
                "mention": mention,
                "gold_gene_ids": "|".join(sorted(set(gids))),
            })

    return rows


def load_pred_spans():
    pred = set()

    with open(PRED_FILE, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            pmid, start, end, mention, etype, gid = parts[:6]

            if etype != "Gene":
                continue

            gid = gid.replace("NCBIGene:", "").replace("*", "").strip()

            if gid and gid[0].isdigit():
                pred.add((pmid, int(start), int(end)))

    return pred


gold_rows = load_gold()
pred_spans = load_pred_spans()

missing = []

for row in gold_rows:
    key = (row["pmid"], row["start"], row["end"])
    if key not in pred_spans:
        missing.append(row)

with open(OUT, "w", encoding="utf-8") as out:
    out.write("pmid\tstart\tend\tmention\tgold_gene_ids\n")
    for row in missing:
        out.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                row["pmid"],
                row["start"],
                row["end"],
                row["mention"],
                row["gold_gene_ids"],
            )
        )

print("saved:", OUT)
print("missing:", len(missing))
