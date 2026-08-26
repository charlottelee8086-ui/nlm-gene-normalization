import csv
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

GOLD_ARROW = Path("nlm_gene-test.arrow")
GNORM_PRED = Path("/mnt/beegfs/home/xli/nlm_gene_repro/GNorm2/output/nlm_gene_test.PubTator")
V6 = Path("family_reranker_predictions_v6.tsv")


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
    gold = {}
    mention_text = {}

    for doc in read_arrow(GOLD_ARROW):
        pmid = str(doc["document_id"])

        for ent in doc["entities"]:
            if ent["type"] not in {"Gene", "GENERIF", "STARGENE"}:
                continue

            if not ent.get("normalized"):
                continue

            start, end = ent["offsets"][0]
            mention = ent["text"][0].strip()

            gids = set()
            for norm in ent["normalized"]:
                if norm.get("db_name") == "NCBIGene":
                    gids.add(str(norm["db_id"]))

            if gids:
                key = (pmid, int(start), int(end))
                gold[key] = gids
                mention_text[key] = mention

    return gold, mention_text


def load_gnorm_pred():
    pred = {}

    with open(GNORM_PRED, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 6:
                continue

            pmid, start, end, mention, etype, gid = parts[:6]

            if etype != "Gene":
                continue

            gid = clean_gid(gid)

            if gid and gid[0].isdigit():
                pred[(pmid, int(start), int(end))] = gid

    return pred


def load_v6_by_pmid_mention_gold():
    """
    v6 file does not contain offsets.
    Use (pmid, mention, gold_gene_ids) as a soft key.
    This is enough for checking whether a recognized GNormPlus mention
    is one of the family cases corrected by v6.
    """
    v6 = {}

    with open(V6, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            key = (
                row["pmid"],
                row["mention"],
                "|".join(sorted(row["gold_gene_ids"].split("|"))),
            )
            v6[key] = row["v6_pred_gid"]

    return v6


gold, mention_text = load_gold()
gnorm_pred = load_gnorm_pred()
v6 = load_v6_by_pmid_mention_gold()

recognized = 0

gnorm_correct = 0
v6_correct = 0

v6_used = 0
gain = 0
hurt = 0

for key, gold_gids in gold.items():
    pmid, start, end = key

    # BELB-style filtered:
    # only evaluate mentions recognized by GNormPlus
    if key not in gnorm_pred:
        continue

    recognized += 1

    mention = mention_text[key]
    gold_key = "|".join(sorted(gold_gids))

    gnorm_gid = gnorm_pred[key]
    gnorm_ok = gnorm_gid in gold_gids

    v6_key = (pmid, mention, gold_key)
    final_gid = v6.get(v6_key, gnorm_gid)

    if v6_key in v6:
        v6_used += 1

    v6_ok = final_gid in gold_gids

    gnorm_correct += int(gnorm_ok)
    v6_correct += int(v6_ok)

    if (not gnorm_ok) and v6_ok:
        gain += 1

    if gnorm_ok and (not v6_ok):
        hurt += 1

print("recognized denominator:", recognized)
print("GNormPlus correct:", gnorm_correct)
print("GNormPlus BELB-style acc:", gnorm_correct / recognized)

print()
print("v6 used on recognized mentions:", v6_used)
print("v6 correct:", v6_correct)
print("v6 BELB-style acc:", v6_correct / recognized)

print()
print("gain:", gain)
print("hurt:", hurt)
print("net gain:", gain - hurt)
