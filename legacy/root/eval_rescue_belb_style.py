import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path


GOLD_ARROW = Path("nlm_gene-test.arrow")
PRED = Path.home() / "nlm_gene_repro/GNorm2/output/nlm_gene_test.PubTator"
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

            if gids:
                key = (pmid, int(start), int(end))
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
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 2:
                continue

            mention = parts[0]
            gid = clean_gid(parts[1])

            if gid and gid[0].isdigit():
                rescue[mention] = gid

    return rescue


gold, mention_text = load_gold()
pred = load_pred()
rescue = load_rescue()

recognized = 0

base_correct = 0
base_wrong = 0

rescue_correct = 0
rescue_wrong = 0

changed_by_rescue = 0
gain = 0
hurt = 0

for key, gold_gids in gold.items():

    # BELB-style filtered evaluation:
    # only evaluate gold mentions recognized by GNormPlus.
    if key not in pred:
        continue

    recognized += 1

    mention = mention_text[key]

    base_gid = pred[key]
    base_ok = base_gid in gold_gids

    if base_ok:
        base_correct += 1
    else:
        base_wrong += 1

    final_gid = base_gid

    # Apply rescue only if this mention string is in rescue dictionary.
    if mention in rescue:
        final_gid = rescue[mention]

    final_ok = final_gid in gold_gids

    if final_gid != base_gid:
        changed_by_rescue += 1

    if final_ok:
        rescue_correct += 1
    else:
        rescue_wrong += 1

    if (not base_ok) and final_ok:
        gain += 1

    if base_ok and (not final_ok):
        hurt += 1


print("Gold mentions:", len(gold))
print("Recognized denominator:", recognized)

print("\n=== Base GNormPlus BELB-style ===")
print("Correct:", base_correct)
print("Wrong:", base_wrong)
print("Accuracy:", base_correct / recognized if recognized else 0)

print("\n=== Rescue BELB-style ===")
print("Correct:", rescue_correct)
print("Wrong:", rescue_wrong)
print("Accuracy:", rescue_correct / recognized if recognized else 0)

print("\n=== Rescue effect on recognized mentions ===")
print("Changed by rescue:", changed_by_rescue)
print("Gain:", gain)
print("Hurt:", hurt)
print("Net gain:", gain - hurt)
