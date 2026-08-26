import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

GOLD_ARROW = Path("nlm_gene-test.arrow")
PRED = Path.home() / "nlm_gene_repro/GNorm2/gold_norm_output/nlm_gene_test.PubTator"
RESCUE = Path("rescue_dictionary.tsv")


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

            mention = ent["text"][0].strip()
            start, end = ent["offsets"][0]

            gids = set()
            for norm in ent["normalized"]:
                if norm.get("db_name") == "NCBIGene":
                    gids.add(str(norm["db_id"]))

            key = (pmid, int(start), int(end))
            if gids:
                gold[key] = gids
                mention_text[key] = mention

    return gold, mention_text


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
            if gid and gid[0].isdigit():
                pred[(pmid, int(start), int(end))] = gid

    return pred


def load_rescue():
    rescue = {}

    with open(RESCUE, encoding="utf-8") as f:
        next(f)
        for line in f:
            mention, gid, freq, ratio = line.rstrip("\n").split("\t")
            rescue[mention] = gid

    return rescue


gold, mention_text = load_gold()
pred = load_pred()
rescue = load_rescue()

base_correct = 0
base_wrong = 0
base_missing = 0

rescued_correct = 0
rescued_wrong = 0
still_missing = 0

final_correct = 0
final_wrong = 0
final_missing = 0

for key, gold_gids in gold.items():
    if key in pred:
        if pred[key] in gold_gids:
            base_correct += 1
            final_correct += 1
        else:
            base_wrong += 1
            final_wrong += 1
    else:
        base_missing += 1
        mention = mention_text[key]

        if mention in rescue:
            gid = rescue[mention]
            if gid in gold_gids:
                rescued_correct += 1
                final_correct += 1
            else:
                rescued_wrong += 1
                final_wrong += 1
        else:
            still_missing += 1
            final_missing += 1

total = len(gold)

print("Gold mentions:", total)

print("\n=== Base GNormPlus ===")
print("Correct:", base_correct)
print("Wrong:", base_wrong)
print("Missing:", base_missing)
print("Accuracy all:", base_correct / total)

print("\n=== Rescue Added ===")
print("Rescued correct:", rescued_correct)
print("Rescued wrong:", rescued_wrong)
print("Still missing:", still_missing)

print("\n=== Final ===")
print("Correct:", final_correct)
print("Wrong:", final_wrong)
print("Missing:", final_missing)
print("Accuracy all:", final_correct / total)
print("Accuracy normalized only:", final_correct / (final_correct + final_wrong))
