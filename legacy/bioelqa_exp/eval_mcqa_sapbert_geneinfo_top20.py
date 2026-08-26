# -*- coding: utf-8 -*-

import re
import string
from collections import Counter
import pandas as pd

GOLD = "bioelqa_dev_mcqa_gold_sapbert_geneinfo_top20.tsv"
PRED = "bioelqa_dev_mcqa_predictions_qwen14_sapbert_geneinfo_top20.txt"

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
            cands.append(parts[0])
    case2cands[cid] = cands

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cid = parts[0]

        # accept formats like:
        # dev_sapbert_case_1 Answer: M
        # dev_sapbert_case_1 M
        m = re.search(r"Answer:\s*([A-Z])", line)
        if not m:
            m = re.search(r"\b([A-Z])\b", line)

        if m:
            pred[cid] = m.group(1)

total = len(gold)
evaluated = 0
valid = 0
correct = 0
gold_in_candidates = 0

invalid = Counter()
wrong_chosen = Counter()

for cid, gold_gids in gold.items():
    cands = case2cands[cid]

    if gold_gids & set(cands):
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

    valid += 1
    pred_gid = cands[idx]

    if pred_gid in gold_gids:
        correct += 1
    else:
        wrong_chosen[pred_gid] += 1

print("=== MCQA DEV Evaluation ===")
print("total prompts:", total)
print("evaluated predictions:", evaluated)
print("valid option predictions:", valid)
print("gold in candidates:", gold_in_candidates, f"{gold_in_candidates/total:.2%}")
print("correct:", correct)
print("accuracy on prompts:", correct / total if total else 0)
print("accuracy on evaluated:", correct / evaluated if evaluated else 0)
print("accuracy on valid options:", correct / valid if valid else 0)

print("\nInvalid answers:")
for k, v in invalid.most_common(20):
    print(k, v)

print("\nTop wrong chosen:")
for k, v in wrong_chosen.most_common(20):
    print(k, v)
