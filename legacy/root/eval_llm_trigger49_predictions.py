import re

GOLD = "llm_trigger49_gold.tsv"
PRED = "llm_trigger49_predictions.txt"

gold = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gids = line.rstrip("\n").split("\t")
        gold[case_id] = set(gids.split("|"))

correct = 0
total = 0

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if not m_case or not m_gid:
            if line.strip():
                print("cannot parse:", line.strip())
            continue

        case_id = m_case.group(1)
        pred_gid = m_gid.group(1)

        if case_id not in gold:
            continue

        ok = pred_gid in gold[case_id]
        total += 1
        correct += int(ok)

        print(case_id, "pred=", pred_gid, "gold=", "|".join(sorted(gold[case_id])), "correct=", ok)

print("total:", total)
print("correct:", correct)
print("acc:", correct / total if total else 0)
