import re

GOLD = "auto_family_llm_gold.tsv"
PRED = "auto_family_llm_predictions_qwen14.txt"

gold = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gold_gene_ids, current_pred_gid = line.rstrip("\n").split("\t")
        gold[case_id] = {
            "mention": mention,
            "gold": set(gold_gene_ids.split("|")),
            "current_pred_gid": current_pred_gid,
        }

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(auto_family_case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if not m_case or not m_gid:
            if line.strip():
                print("cannot parse:", line.strip())
            continue

        pred[m_case.group(1)] = m_gid.group(1)

total = 0
llm_correct = 0
current_correct = 0
gain = 0
hurt = 0

print("case_id\tmention\tcurrent_pred\tllm_pred\tgold\tcurrent_ok\tllm_ok")

for case_id, ex in gold.items():
    total += 1
    llm_gid = pred.get(case_id, "missing")

    current_ok = ex["current_pred_gid"] in ex["gold"]
    llm_ok = llm_gid in ex["gold"]

    current_correct += int(current_ok)
    llm_correct += int(llm_ok)

    if not current_ok and llm_ok:
        gain += 1
    if current_ok and not llm_ok:
        hurt += 1

    print(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            case_id,
            ex["mention"],
            ex["current_pred_gid"],
            llm_gid,
            "|".join(sorted(ex["gold"])),
            int(current_ok),
            int(llm_ok),
        )
    )

print()
print("total:", total)
print("current correct:", current_correct)
print("LLM correct:", llm_correct)
print("LLM acc:", llm_correct / total if total else 0)
print("gain:", gain)
print("hurt:", hurt)
print("net gain:", gain - hurt)
