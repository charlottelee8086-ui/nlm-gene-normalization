import csv
import re

BASE = "family_reranker_predictions_v5_llm_species.tsv"
GOLD = "auto_family_llm_gold.tsv"
PRED = "auto_family_llm_predictions.txt"

TOTAL_GOLD = 2729
BASE_SYSTEM_CORRECT = 1705

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(auto_family_case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if m_case and m_gid:
            pred[m_case.group(1)] = m_gid.group(1)

lookup = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gold_gene_ids, current_pred_gid = line.rstrip("\n").split("\t")
        llm_gid = pred.get(case_id)

        if not llm_gid:
            continue

        key = (
            pmid,
            mention,
            "|".join(sorted(gold_gene_ids.split("|"))),
        )

        lookup[key] = llm_gid

base_correct = 0
v7_correct = 0
total = 0
oracle = 0
used_llm = 0
gain = 0
hurt = 0

with open(BASE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        total += 1
        gold = set(row["gold_gene_ids"].split("|"))

        base_ok = row["correct"] == "1"
        base_correct += int(base_ok)

        if row["oracle"] == "1":
            oracle += 1

        key = (
            row["pmid"],
            row["mention"],
            "|".join(sorted(row["gold_gene_ids"].split("|"))),
        )

        llm_gid = lookup.get(key)

        if llm_gid:
            used_llm += 1
            final_ok = llm_gid in gold
        else:
            final_ok = base_ok

        v7_correct += int(final_ok)

        if (not base_ok) and final_ok:
            gain += 1

        if base_ok and not final_ok:
            hurt += 1

print("family cases:", total)
print("oracle:", oracle)
print("v5 correct:", base_correct)
print("v7 correct:", v7_correct)
print("LLM used:", used_llm)
print("gain:", gain)
print("hurt:", hurt)
print("net gain:", gain - hurt)
print("v5 family accuracy:", base_correct / total if total else 0)
print("v7 family accuracy:", v7_correct / total if total else 0)

v5_overall_correct = BASE_SYSTEM_CORRECT + base_correct
v7_overall_correct = BASE_SYSTEM_CORRECT + v7_correct

print()
print("v5 overall correct:", v5_overall_correct)
print("v5 overall accuracy:", v5_overall_correct / TOTAL_GOLD)
print("v7 overall correct:", v7_overall_correct)
print("v7 overall accuracy:", v7_overall_correct / TOTAL_GOLD)
