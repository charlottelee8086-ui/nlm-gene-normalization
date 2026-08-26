# -*- coding: utf-8 -*-

import re
from collections import defaultdict, Counter

GOLD = "direct_llm_fulltest_gold.tsv"
SYMBOL_PRED = "direct_llm_symbol_predictions_qwen14.txt"

# 你已有的训练集 dictionary
TRAIN_KB = "train_synonym_kb.tsv"
GNORM_KB = "gnormplus_synonym_kb_filtered.tsv"

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
        m_sym = re.search(r"Symbol:\s*([A-Za-z0-9_.\-\/]+)", line)
        if m_case:
            symbol_pred[m_case.group(1)] = m_sym.group(1) if m_sym else "NONE"


symbol2gid_counter = defaultdict(Counter)

# 兼容不同tsv格式：只要每行里有 mention/symbol 和 gid，就尽量读
for path in [TRAIN_KB, GNORM_KB]:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.rstrip("\n").split("\t")

                if len(parts) < 2:
                    continue

                # 常见格式：
                # mention \t gene_id ...
                mention = parts[0].strip()
                gid = parts[1].strip()

                gid = gid.replace("NCBIGene:", "").replace("*", "").strip()

                if not gid or not gid[0].isdigit():
                    continue

                key = mention.upper()
                symbol2gid_counter[key][gid] += 1

    except FileNotFoundError:
        print("missing:", path)


def map_symbol_to_gid(sym):
    if not sym or sym == "NONE":
        return "NONE"

    key = sym.strip().upper()

    if key not in symbol2gid_counter:
        return "NONE"

    return symbol2gid_counter[key].most_common(1)[0][0]


total = len(gold)
done = 0
mapped = 0
correct = 0

for cid, gids in gold.items():
    if cid not in symbol_pred:
        continue

    done += 1
    sym = symbol_pred[cid]
    gid = map_symbol_to_gid(sym)

    if gid != "NONE":
        mapped += 1

    if gid in gids:
        correct += 1

print("total gold:", total)
print("evaluated:", done)
print("symbol mapped:", mapped)
print("correct:", correct)
print("accuracy on evaluated:", correct / done if done else 0)
print("mapping coverage:", mapped / done if done else 0)
