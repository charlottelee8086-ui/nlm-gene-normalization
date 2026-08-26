import re

LLM = "llm_trigger49_predictions.txt"
GOLD = "llm_trigger49_gold.tsv"
BERT = "family_reranker_predictions_v4_species.tsv"

gold = {}
mention = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, m, gids = line.rstrip("\n").split("\t")
        gold[case_id] = set(gids.split("|"))
        mention[case_id] = m

llm_pred = {}

with open(LLM, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)
        if m_case and m_gid:
            llm_pred[m_case.group(1)] = m_gid.group(1)

# Need map trigger cases in order from exported gold to BERT rows by mention/gold.
bert_rows = []
with open(BERT, encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        pmid = parts[0]
        m = parts[1]
        pred_gid = parts[3]
        gold_gids = set(parts[6].split("|"))
        bert_correct = parts[11] == "1"
        bert_rows.append((pmid, m, pred_gid, gold_gids, bert_correct))

bert_used = []
used = set()

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, m, gids = line.rstrip("\n").split("\t")
        gold_set = set(gids.split("|"))

        found = None
        for i, row in enumerate(bert_rows):
            if i in used:
                continue
            b_pmid, b_m, b_pred, b_gold, b_ok = row
            if b_pmid == pmid and b_m == m and b_gold == gold_set:
                found = (i, row)
                break

        if found is None:
            print("not found:", case_id, pmid, m, gids)
            continue

        i, row = found
        used.add(i)

        llm_gid = llm_pred.get(case_id)
        llm_ok = llm_gid in gold_set if llm_gid else False
        bert_ok = row[4]

        bert_used.append((case_id, m, bert_ok, llm_ok, llm_gid, gold_set))

bert_correct = sum(1 for x in bert_used if x[2])
llm_correct = sum(1 for x in bert_used if x[3])

both_correct = sum(1 for x in bert_used if x[2] and x[3])
llm_gain = sum(1 for x in bert_used if (not x[2]) and x[3])
llm_hurt = sum(1 for x in bert_used if x[2] and (not x[3]))

print("cases:", len(bert_used))
print("PubMedBERT correct:", bert_correct)
print("LLM correct:", llm_correct)
print("Both correct:", both_correct)
print("LLM gains over BERT:", llm_gain)
print("LLM hurts BERT:", llm_hurt)
print("Net gain:", llm_gain - llm_hurt)

BASE_TOTAL_CORRECT = 1781  # 1705 + 76, current best with species-aware reranker
TOTAL_GOLD = 2729

new_correct = BASE_TOTAL_CORRECT + llm_gain - llm_hurt
print("New estimated correct:", new_correct)
print("New estimated accuracy:", new_correct / TOTAL_GOLD)
