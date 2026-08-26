# -*- coding: utf-8 -*-

import re
from collections import Counter, defaultdict

GOLD = "bioelqa_dev_symbol_species_gold.tsv"
PRED = "bioelqa_dev_symbol_species_predictions_qwen14.txt"
NCBI_KB = "ncbi_symbol_synonym_taxid_kb.tsv"

SPECIES2TAX = {
    "human": "9606",
    "mouse": "10090",
    "murine": "10090",
    "rat": "10116",
    "zebrafish": "7955",
    "fruit fly": "7227",
    "drosophila": "7227",
    "arabidopsis": "3702",
    "worm": "6239",
    "c. elegans": "6239",
    "yeast": "4932",
}

PRIOR_TAX = ["9606", "10090", "10116"]


def normalize_symbol(sym):
    sym = str(sym).strip()
    sym = sym.replace("Gene Symbol:", "").replace("Symbol:", "").strip()
    sym = sym.strip("`'\".,;:()[]{}")
    sym = sym.replace("β", "BETA").replace("α", "ALPHA").replace("γ", "GAMMA")
    return sym.upper()


def normalize_species(sp):
    sp = str(sp).strip().lower()
    sp = sp.replace("species:", "").strip()
    sp = sp.strip("`'\".,;:()[]{}")
    return sp


def read_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            case_id, doc_id, mention, gids = line.rstrip("\n").split("\t")
            gold[case_id] = set(gids.split("|"))
    return gold


def read_predictions(path):
    pred = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m_case = re.search(r"(dev_symbol_species_case_\d+)", line)
            if not m_case:
                continue

            cid = m_case.group(1)

            m_sym = re.search(r"Symbol:\s*([^\t\n\r]+)", line)
            m_species = re.search(r"Species:\s*([^\t\n\r]+)", line)

            sym = normalize_symbol(m_sym.group(1)) if m_sym else "NONE"
            species = normalize_species(m_species.group(1)) if m_species else "unclear"

            pred[cid] = (sym, species)

    return pred


def load_ncbi_kb(path):
    term_taxid2gid = {}
    term2all = defaultdict(list)

    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            term, tax_id, gid, count = parts[:4]
            key = normalize_symbol(term)

            try:
                count = int(count)
            except Exception:
                count = 0

            # exact symbol/synonym + taxid lookup
            if (key, tax_id) not in term_taxid2gid:
                term_taxid2gid[(key, tax_id)] = gid

            term2all[key].append((tax_id, gid, count))

    return term_taxid2gid, term2all


def map_symbol_species_to_gid(sym, species, term_taxid2gid, term2all):
    if not sym or sym == "NONE":
        return "NONE", "none"

    taxid = SPECIES2TAX.get(species)

    # 1. true species-aware exact lookup
    if taxid:
        gid = term_taxid2gid.get((sym, taxid))
        if gid:
            return gid, f"exact_species_{taxid}"

    # 2. if species unclear, but symbol globally maps to one GeneID
    if sym in term2all:
        candidates = term2all[sym]
        unique_gids = sorted(set(gid for taxid, gid, count in candidates))

        if len(unique_gids) == 1:
            return unique_gids[0], "unique_global"

        # 3. fallback only when exact species failed or species unclear
        for tax in PRIOR_TAX:
            tax_candidates = [(gid, count) for taxid, gid, count in candidates if taxid == tax]
            if tax_candidates:
                tax_candidates = sorted(tax_candidates, key=lambda x: -x[1])
                return tax_candidates[0][0], f"prior_tax_{tax}"

        # 4. final fallback by count
        candidates = sorted(candidates, key=lambda x: -x[2])
        return candidates[0][1], "count_fallback"

    return "NONE", "unmapped"


gold = read_gold(GOLD)
pred = read_predictions(PRED)

print("loaded gold:", len(gold))
print("loaded predictions:", len(pred))

print("loading NCBI KB...")
term_taxid2gid, term2all = load_ncbi_kb(NCBI_KB)
print("loaded KB terms:", len(term2all))

total = len(gold)
done = 0
mapped = 0
correct = 0

mode_counter = Counter()
species_counter = Counter()
unmapped = Counter()
wrong = Counter()

for cid, gids in gold.items():
    if cid not in pred:
        continue

    done += 1
    sym, species = pred[cid]
    species_counter[species] += 1

    gid, mode = map_symbol_species_to_gid(sym, species, term_taxid2gid, term2all)
    mode_counter[mode] += 1

    if gid != "NONE":
        mapped += 1
    else:
        unmapped[(sym, species)] += 1

    if gid in gids:
        correct += 1
    else:
        wrong[(sym, species, gid, "|".join(sorted(gids)), mode)] += 1

print("\n=== DEV Symbol + Species → GeneID Evaluation ===")
print("total gold:", total)
print("evaluated:", done)
print("symbol mapped:", mapped)
print("correct:", correct)
print("accuracy on evaluated:", correct / done if done else 0)
print("mapping coverage:", mapped / done if done else 0)

print("\nPredicted species distribution:")
for k, v in species_counter.most_common():
    print(k, v)

print("\nMapping modes:")
for k, v in mode_counter.most_common():
    print(k, v)

print("\nTop unmapped:")
for (sym, sp), cnt in unmapped.most_common(30):
    print(cnt, sym, sp)

print("\nTop wrong mapped:")
for (sym, sp, gid, gold_gids, mode), cnt in wrong.most_common(30):
    print(cnt, sym, sp, gid, gold_gids, mode)
