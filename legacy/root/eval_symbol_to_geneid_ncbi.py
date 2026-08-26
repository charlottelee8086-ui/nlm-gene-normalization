# -*- coding: utf-8 -*-

import re
from collections import defaultdict, Counter

GOLD = "direct_llm_fulltest_gold.tsv"
SYMBOL_PRED = "direct_llm_symbol_predictions_qwen14.txt"
NCBI_KB = "ncbi_symbol_synonym_kb.tsv"

gold = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gids = line.rstrip("\n").split("\t")
        gold[case_id] = set(gids.split("|"))

symbol_pred = {}

with open(SYMBOL_PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(direct_case_\d+)", line)
        m_sym = re.search(r"Symbol:\s*(.+)", line)
        if m_case:
            sym = m_sym.group(1).strip() if m_sym else "NONE"
            symbol_pred[m_case.group(1)] = sym

term2gid = {}

with open(NCBI_KB, encoding="utf-8") as f:
    next(f)
    for line in f:
        term, gid, count = line.rstrip("\n").split("\t")
        term2gid[term.upper()] = gid


def normalize_symbol(sym):
    sym = sym.strip()
    sym = sym.replace("Gene Symbol:", "").strip()
    sym = sym.replace("Symbol:", "").strip()
    sym = sym.strip("`'\".,;:()[]{}")
    return sym.upper()


def map_symbol_to_gid(sym):
    if not sym or sym == "NONE":
        return "NONE"

    key = normalize_symbol(sym)

    if key in term2gid:
        return term2gid[key]

    return "NONE"


total = len(gold)
done = 0
mapped = 0
correct = 0
unmapped = Counter()
wrong = Counter()

for cid, gids in gold.items():
    if cid not in symbol_pred:
        continue

    done += 1
    sym = symbol_pred[cid]
    gid = map_symbol_to_gid(sym)

    if gid != "NONE":
        mapped += 1
    else:
        unmapped[normalize_symbol(sym)] += 1

    if gid in gids:
        correct += 1
    else:
        wrong[(normalize_symbol(sym), gid)] += 1

print("total gold:", total)
print("evaluated:", done)
print("symbol mapped:", mapped)
print("correct:", correct)
print("accuracy on evaluated:", correct / done if done else 0)
print("mapping coverage:", mapped / done if done else 0)

print("\nTop unmapped symbols:")
for sym, cnt in unmapped.most_common(30):
    print(cnt, sym)
