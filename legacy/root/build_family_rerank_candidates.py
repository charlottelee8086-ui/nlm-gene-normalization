import json
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

ARROW = Path("nlm_gene-test.arrow")
STILL = Path("still_missing_after_rescue.tsv")
OUT = Path("family_rerank_candidates.jsonl")

CANDIDATES = {
    "mapk": {"5594", "5595", "26413", "26416", "26417", "50689", "1432"},
    "mitogenactivatedproteinkinase": {"5594", "5595", "26413", "26416", "26417", "50689", "1432"},
    "erk12": {"5594", "5595", "26417"},
    "erk1and2": {"5594", "5595", "26417"},
    "mek12": {"5604", "5605", "26395", "26396"},
    "nfkb": {"4790", "4791", "5970", "309165", "18033", "81736"},
    "nfkappab": {"4790", "4791", "5970", "309165", "18033", "81736"},
    "hif1alpha": {"3091", "29560"},
    "hif1a": {"3091", "29560"},
    "wnt": {"7471", "7472", "7473", "7474", "7475", "22408", "22410", "22415"},
    "notch": {"4851", "4853", "18128", "18129", "18130", "18131", "30718"},
    "stat": {"6772", "6773", "6774", "6775", "20848", "20850"},
    "akt": {"207", "208", "10000", "11651", "11652", "23797", "101910198"},
    "smad": {"4086", "4087", "4088", "4089", "4090", "17125", "17126", "17127", "17128", "17129", "55994"},
    "mcp1": {"6347", "24770", "20296"},
    "ccl2": {"6347", "24770", "20296"},
    "cxcl9": {"17329", "4283"},
    "cxcl10": {"15945", "3627"},
    "cxcl11": {"56066"},
    "chemokine": {"6347", "24770", "20296", "17329", "4283", "15945", "3627", "56066", "6352", "6387"},
    "chemokines": {"6347", "24770", "20296", "17329", "4283", "15945", "3627", "56066", "6352", "6387"},
    "h3": {"15268", "838577"},
    "histone": {"15268", "838577", "180073", "837440"},
    "cd68": {"968", "12514", "287435"},
    "cd44": {"960", "12505"},
    "cd14": {"929", "12475", "60350"},
    "prc2": {"2146", "2147", "2148"},
    "tric": {"6950"},
    "ikbkinase": {"3551", "3552"},
}


def norm(s):
    s = s.lower().strip()
    s = s.replace("κ", "kappa")
    s = s.replace("β", "beta")
    s = s.replace("α", "alpha")
    s = s.replace("γ", "gamma")
    s = s.replace("δ", "delta")
    s = re.sub(r"[\s_\-/]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def get_candidates(mention):
    m = norm(mention)
    cands = set()

    if m in CANDIDATES:
        cands |= CANDIDATES[m]

    if "mapk" in m or "mitogenactivatedproteinkinase" in m:
        cands |= CANDIDATES["mapk"]
    if "erk12" in m or "erk1and2" in m:
        cands |= CANDIDATES["erk12"]
    if "mek12" in m or "mek1and2" in m:
        cands |= CANDIDATES["mek12"]
    if "nfkb" in m or "nfkappab" in m:
        cands |= CANDIDATES["nfkb"]
    if "hif1alpha" in m or "hif1a" in m:
        cands |= CANDIDATES["hif1alpha"]
    if m.startswith("wnt"):
        cands |= CANDIDATES["wnt"]
    if m.startswith("stat"):
        cands |= CANDIDATES["stat"]
    if m.startswith("smad"):
        cands |= CANDIDATES["smad"]
    if m.startswith("akt"):
        cands |= CANDIDATES["akt"]

    return sorted(cands)


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pylist()
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
            return reader.read_all().to_pylist()


def load_texts():
    pmid2text = {}

    for doc in read_arrow(ARROW):
        pmid = str(doc["document_id"])
        passages = sorted(doc["passages"], key=lambda p: p["offsets"][0][0])
        text = " ".join(p["text"][0] for p in passages)
        pmid2text[pmid] = text

    return pmid2text


def get_context(text, start, end, window=300):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


pmid2text = load_texts()

total = 0
with_gold_in_candidates = 0

with open(STILL, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    next(f)

    for line in f:
        pmid, start, end, mention, gids = line.rstrip("\n").split("\t")
        start = int(start)
        end = int(end)

        cands = get_candidates(mention)
        if not cands:
            continue

        gold = gids.split("|")
        text = pmid2text.get(pmid, "")
        context = get_context(text, start, end)

        total += 1
        if set(gold) & set(cands):
            with_gold_in_candidates += 1

        out.write(json.dumps({
            "pmid": pmid,
            "start": start,
            "end": end,
            "mention": mention,
            "context": context,
            "gold_gene_ids": gold,
            "candidate_gene_ids": cands,
            "gold_in_candidates": bool(set(gold) & set(cands)),
        }, ensure_ascii=False) + "\n")

print("saved:", OUT)
print("family candidate cases:", total)
print("gold in candidates:", with_gold_in_candidates)

