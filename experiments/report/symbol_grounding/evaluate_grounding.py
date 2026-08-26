from pathlib import Path
# -*- coding: utf-8 -*-

import re
from collections import Counter, defaultdict

GOLD = "direct_llm_fulltest_gold.tsv"
SYMBOL_PRED = "direct_llm_symbol_predictions_qwen14.txt"
NCBI_KB = "ncbi_symbol_synonym_taxid_kb.tsv"

# GNormPlus species assignment file
TMP_SA = Path(__file__).resolve().parents[3] / "data/external/gnormplus_tmp_SA.PubTator"


def normalize_symbol(sym):
    sym = sym.strip()
    sym = sym.replace("Gene Symbol:", "").strip()
    sym = sym.replace("Symbol:", "").strip()
    sym = sym.strip("`'\".,;:()[]{}")
    return sym.upper()


def load_focus_taxid():
    pmid2taxid = {}

    with open(TMP_SA, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\tSpecies\t*" not in line:
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            pmid = parts[0]
            taxid = parts[-1].replace("*", "").strip()

            if taxid:
                pmid2taxid[pmid] = taxid

    return pmid2taxid


gold = {}
case2pmid = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gids = line.rstrip("\n").split("\t")
        gold[case_id] = set(gids.split("|"))
        case2pmid[case_id] = pmid


symbol_pred = {}

with open(SYMBOL_PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(direct_case_\d+)", line)
        m_sym = re.search(r"Symbol:\s*(.+)", line)

        if m_case:
            sym = m_sym.group(1).strip() if m_sym else "NONE"
            symbol_pred[m_case.group(1)] = normalize_symbol(sym)


term_taxid2gid = {}
term2all = defaultdict(list)

with open(NCBI_KB, encoding="utf-8") as f:
    next(f)
    for line in f:
        term, tax_id, gid, count = line.rstrip("\n").split("\t")
        key = term.upper()

        term_taxid2gid[(key, tax_id)] = gid
        term2all[key].append((tax_id, gid))


pmid2taxid = load_focus_taxid()


def map_symbol_to_gid(sym, focus_taxid):
    if not sym or sym == "NONE":
        return "NONE", "none"

    # first try exact symbol + focus species
    if focus_taxid and (sym, focus_taxid) in term_taxid2gid:
        return term_taxid2gid[(sym, focus_taxid)], "species"

    # fallback: if only one gene exists for this symbol globally
    if sym in term2all:
        candidates = term2all[sym]
        unique_gids = sorted(set(gid for taxid, gid in candidates))

        if len(unique_gids) == 1:
            return unique_gids[0], "unique_global"

        # fallback: prefer human if available
        for taxid, gid in candidates:
            if taxid == "9606":
                return gid, "human_fallback"

        return candidates[0][1], "first_fallback"

    return "NONE", "unmapped"


total = len(gold)
done = 0
mapped = 0
correct = 0

mode_counter = Counter()
unmapped = Counter()
wrong = Counter()

for cid, gids in gold.items():
    if cid not in symbol_pred:
        continue

    done += 1

    pmid = case2pmid[cid]
    focus_taxid = pmid2taxid.get(pmid, "")

    sym = symbol_pred[cid]
    gid, mode = map_symbol_to_gid(sym, focus_taxid)

    mode_counter[mode] += 1

    if gid != "NONE":
        mapped += 1
    else:
        unmapped[sym] += 1

    if gid in gids:
        correct += 1
    else:
        wrong[(sym, gid, focus_taxid)] += 1

print("total gold:", total)
print("evaluated:", done)
print("symbol mapped:", mapped)
print("correct:", correct)
print("accuracy on evaluated:", correct / done if done else 0)
print("mapping coverage:", mapped / done if done else 0)

print("\nMapping modes:")
for k, v in mode_counter.most_common():
    print(k, v)

print("\nTop unmapped symbols:")
for sym, cnt in unmapped.most_common(30):
    print(cnt, sym)

print("\nTop wrong mapped symbols:")
for (sym, gid, taxid), cnt in wrong.most_common(30):
    print(cnt, sym, gid, taxid)
