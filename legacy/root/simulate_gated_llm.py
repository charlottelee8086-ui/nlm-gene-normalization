import re

LLM = "llm_trigger49_predictions.txt"
GOLD = "llm_trigger49_gold.tsv"
BERT = "family_reranker_predictions_v4_species.tsv"

BASE_TOTAL_CORRECT = 1781
TOTAL_GOLD = 2729

gold_rows = []

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gids = line.rstrip("\n").split("\t")
        gold_rows.append({
            "case_id": case_id,
            "pmid": pmid,
            "mention": mention,
            "gold": set(gids.split("|")),
        })

llm_pred = {}

with open(LLM, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)
        if m_case and m_gid:
            llm_pred[m_case.group(1)] = m_gid.group(1)

bert_rows = []

with open(BERT, encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        bert_rows.append({
            "pmid": parts[0],
            "mention": parts[1],
            "focus_taxid": parts[2],
            "bert_pred": parts[3],
            "bert_symbol": parts[4],
            "bert_taxid": parts[5],
            "gold": set(parts[6].split("|")),
            "bert_correct": parts[11] == "1",
        })

used = set()

bert_correct = 0
llm_correct = 0
gated_correct = 0

use_bert = 0
use_llm = 0

gated_gain = 0
gated_hurt = 0

for g in gold_rows:
    found = None

    for i, b in enumerate(bert_rows):
        if i in used:
            continue

        if (
            b["pmid"] == g["pmid"]
            and b["mention"] == g["mention"]
            and b["gold"] == g["gold"]
        ):
            found = (i, b)
            break

    if found is None:
        print("not found:", g)
        continue

    i, b = found
    used.add(i)

    bert_ok = b["bert_correct"]

    llm_gid = llm_pred.get(g["case_id"])
    llm_ok = llm_gid in g["gold"] if llm_gid else False

    # Gate rule:
    # If BERT prediction species matches focus species, trust BERT.
    # Otherwise use LLM.
    focus = b["focus_taxid"]
    bert_tax = b["bert_taxid"]

    if focus and bert_tax == focus:
        final_ok = bert_ok
        use_bert += 1
    else:
        final_ok = llm_ok
        use_llm += 1

    bert_correct += int(bert_ok)
    llm_correct += int(llm_ok)
    gated_correct += int(final_ok)

    if (not bert_ok) and final_ok:
        gated_gain += 1

    if bert_ok and (not final_ok):
        gated_hurt += 1

print("cases:", len(gold_rows))
print("BERT correct:", bert_correct)
print("LLM correct:", llm_correct)
print("Gated correct:", gated_correct)
print("Use BERT:", use_bert)
print("Use LLM:", use_llm)
print("Gated gain over BERT:", gated_gain)
print("Gated hurt over BERT:", gated_hurt)
print("Net gain:", gated_gain - gated_hurt)

new_correct = BASE_TOTAL_CORRECT + gated_gain - gated_hurt

print("New estimated correct:", new_correct)
print("New estimated accuracy:", new_correct / TOTAL_GOLD)
