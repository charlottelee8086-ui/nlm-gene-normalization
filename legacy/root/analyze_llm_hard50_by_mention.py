import re
from collections import defaultdict

gold = {}

with open("llm_hard50_gold.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gid = line.rstrip("\n").split("\t")
        gold[case_id] = (mention, gid)

stats = defaultdict(lambda: [0, 0])

with open("llm_hard50_predictions.txt", encoding="utf-8") as f:
    for line in f:

        m_case = re.search(r"(case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if not m_case or not m_gid:
            continue

        case_id = m_case.group(1)
        pred = m_gid.group(1)

        if case_id not in gold:
            continue

        mention, gold_gid = gold[case_id]

        stats[mention][1] += 1

        if pred == gold_gid:
            stats[mention][0] += 1

print("mention\tcorrect\ttotal\tacc")

for mention, (c, t) in sorted(
        stats.items(),
        key=lambda x: (-x[1][1], x[0])):

    print(
        "{}\t{}\t{}\t{:.3f}".format(
            mention,
            c,
            t,
            c/t
        )
    )
