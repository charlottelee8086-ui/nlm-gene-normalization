# -*- coding: utf-8 -*-

import re
import string
from collections import Counter
import pandas as pd

GOLD = "bioelqa_dev_mcqa_gold_top20_fullabstract.tsv"
PRED = "bioelqa_dev_mcqa_predictions_qwen14_top20_fullabstract.txt"

letters = list(string.ascii_uppercase)

gold_df = pd.read_csv(GOLD, sep="\t")

gold = {}
case2cands = {}

for _, r in gold_df.iterrows():
    cid = str(r["case_id"])
    gold[cid] = set(str(r["gold_geneid"]).split("|"))

    cands = []
    for c in str(r["candidates"]).split("|"):
        parts = c.split("::")
        if len(parts) >= 1:
            cands.append(parts[0])  # GeneID

    case2cands[cid] = cands

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(dev_mcqa_case_\d+)", line)
        m_ans = re.search(r"Answer:\s*([A-Z])", line, re.I)

        if not m_case:
            continue

        cid = m_case.group(1)

        if m_ans:
            ans = m_ans.group(1).upper()
        else:
            m_ans = re.search(r"\b([A-Z])\b", line)
            ans = m_ans.group(1).upper() if m_ans else "NONE"

        pred[cid] = ans

total_gold = len(gold)
evaluated = 0
valid_option = 0
correct = 0
gold_in_candidates = 0
invalid = Counter()
wrong = Counter()

for cid, gids in gold.items():
    cands = case2cands.get(cid, [])

    if gids.intersection(set(cands)):
        gold_in_candidates += 1

    if cid not in pred:
        continue

    evaluated += 1
    ans = pred[cid]

    if ans not in letters:
        invalid[ans] += 1
        continue

    idx = letters.index(ans)

    if idx >= len(cands):
        invalid[ans] += 1
        continue

    valid_option += 1
    chosen_gid = cands[idx]

    if chosen_gid in gids:
        correct += 1
    else:
        wrong[(ans, chosen_gid, "|".join(sorted(gids)))] += 1

print("=== Dictionary Top-20 + Full Abstract DEV Evaluation ===")
print("total prompts:", total_gold)
print("evaluated predictions:", evaluated)
print("valid option predictions:", valid_option)
print("gold in candidates:", gold_in_candidates, f"{gold_in_candidates/total_gold:.2%}")
print("correct:", correct)
print("accuracy on prompts:", correct / total_gold if total_gold else 0)
print("accuracy on evaluated:", correct / evaluated if evaluated else 0)
print("accuracy on valid options:", correct / valid_option if valid_option else 0)

print("\nInvalid answers:")
for k, v in invalid.most_common(20):
    print(k, v)

print("\nTop wrong chosen:")
for (ans, chosen, gids), cnt in wrong.most_common(20):
    print(cnt, "answer=", ans, "chosen_gid=", chosen, "gold=", gids)
