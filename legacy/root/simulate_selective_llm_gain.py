import re
from pathlib import Path

BASE_CORRECT = 1705
TOTAL_GOLD = 2729

PRED = Path("llm_hard50_predictions.txt")
GOLD = Path("llm_hard50_gold.tsv")

GOOD_MENTIONS = {
    "MCP-1",
    "MAPK",
    "HIF-1α",
    "CCL2",
    "CXCL9",
    "CXCL10",
    "NF-κB",
    "MAPK (p38)",
    "mitogen-activated protein kinases",
}

gold = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gid = line.rstrip("\n").split("\t")
        gold[case_id] = {
            "mention": mention,
            "gid": gid,
        }

used = 0
correct = 0

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if not m_case or not m_gid:
            continue

        case_id = m_case.group(1)
        pred_gid = m_gid.group(1)

        if case_id not in gold:
            continue

        mention = gold[case_id]["mention"]
        gold_gid = gold[case_id]["gid"]

        if mention not in GOOD_MENTIONS:
            continue

        used += 1
        if pred_gid == gold_gid:
            correct += 1

print("Selective LLM mentions used:", used)
print("Selective LLM correct:", correct)
print("Selective LLM acc:", correct / used if used else 0)

new_correct = BASE_CORRECT + correct
print("Estimated total correct:", new_correct)
print("Estimated total accuracy:", new_correct / TOTAL_GOLD)
