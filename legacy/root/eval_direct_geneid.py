# -*- coding: utf-8 -*-

import re

GOLD = "direct_llm_fulltest_gold.tsv"
PRED = "direct_llm_geneid_predictions_qwen14.txt"

gold = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gids = line.rstrip("\n").split("\t")
        gold[case_id] = set(gids.split("|"))

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(direct_case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*([0-9]+)", line)
        if m_case:
            pred[m_case.group(1)] = m_gid.group(1) if m_gid else "NONE"

total = len(gold)
done = 0
correct = 0

for cid, gids in gold.items():
    if cid not in pred:
        continue
    done += 1
    if pred[cid] in gids:
        correct += 1

print("total gold:", total)
print("evaluated:", done)
print("correct:", correct)
print("accuracy:", correct / done if done else 0)
