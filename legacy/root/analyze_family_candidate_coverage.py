import re
from collections import Counter

IN = "still_missing_after_rescue.tsv"

# Manual high-value family / complex candidate sets.
# These are NCBI Gene IDs observed in your gold data and common gene-family mappings.
CANDIDATES = {
    # MAPK / ERK / MEK families
    "mapk": {"5594", "5595", "26413", "26416", "26417", "50689", "1432"},
    "mitogenactivatedproteinkinase": {"5594", "5595", "26413", "26416", "26417", "50689", "1432"},
    "erk12": {"5594", "5595", "26417"},
    "erk1and2": {"5594", "5595", "26417"},
    "mek12": {"5604", "5605", "26395", "26396"},

    # NF-kB
    "nfkb": {"4790", "4791", "5970", "309165", "18033", "81736"},
    "nfkappab": {"4790", "4791", "5970", "309165", "18033", "81736"},

    # HIF
    "hif1alpha": {"3091", "29560"},
    "hif1a": {"3091", "29560"},

    # WNT / Notch / STAT / AKT / SMAD broad candidates
    "wnt": {"7471", "7472", "7473", "7474", "7475", "22408", "22410", "22415"},
    "notch": {"4851", "4853", "18128", "18129", "18130", "18131", "30718"},
    "stat": {"6772", "6773", "6774", "6775", "20848", "20850"},
    "akt": {"207", "208", "10000", "11651", "11652", "23797", "101910198"},
    "smad": {"4086", "4087", "4088", "4089", "4090", "17125", "17126", "17127", "17128", "17129", "55994"},

    # Chemokines / cytokines
    "mcp1": {"6347", "24770", "20296"},
    "ccl2": {"6347", "24770", "20296"},
    "cxcl9": {"17329", "4283"},
    "cxcl10": {"15945", "3627"},
    "cxcl11": {"56066"},
    "chemokine": {"6347", "24770", "20296", "17329", "4283", "15945", "3627", "56066", "6352", "6387"},
    "chemokines": {"6347", "24770", "20296", "17329", "4283", "15945", "3627", "56066", "6352", "6387"},

    # Histone
    "h3": {"15268", "838577"},
    "histone": {"15268", "838577", "180073", "837440"},

    # Common markers still ambiguous
    "cd68": {"968", "12514", "287435"},
    "cd44": {"960", "12505"},
    "cd14": {"929", "12475", "60350"},

    # Complexes / other
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
    s = s.replace("β", "beta")
    s = s.replace("κ", "kappa")
    s = s.replace("nf-κb", "nfkappab")
    s = s.replace("nf-kappab", "nfkappab")
    s = s.replace("nf-kappa b", "nfkappab")
    s = re.sub(r"[\s_\-/]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def get_candidates(mention):
    m = norm(mention)

    cands = set()

    # exact normalized lookup
    if m in CANDIDATES:
        cands |= CANDIDATES[m]

    # pattern rules
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

    return cands


rows = []
covered = 0
has_candidates = 0
total = 0

mention_counter = Counter()
covered_counter = Counter()

with open(IN, encoding="utf-8") as f:
    header = next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        pmid, start, end, mention, gids = parts[:5]
        gold = set(gids.split("|"))
        cands = get_candidates(mention)

        total += 1

        if cands:
            has_candidates += 1
            mention_counter[mention] += 1

        if gold & cands:
            covered += 1
            covered_counter[mention] += 1

        rows.append((mention, gids, cands, bool(gold & cands)))

print("Total still missing:", total)
print("Mentions with family candidates:", has_candidates)
print("Gold covered by candidates:", covered)
print("Candidate coverage over all still missing:", covered / total if total else 0)
print("Candidate coverage where candidates exist:", covered / has_candidates if has_candidates else 0)

print("\n=== Top candidate-covered mentions ===\n")
for mention, freq in covered_counter.most_common(50):
    print(freq, mention)

print("\n=== Top mentions with candidates but not always covered ===\n")
for mention, freq in mention_counter.most_common(50):
    cov = covered_counter[mention]
    print(mention, "total=", freq, "covered=", cov)
