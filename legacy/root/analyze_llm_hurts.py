import re

LLM = "llm_trigger49_predictions.txt"
GOLD = "llm_trigger49_gold.tsv"
BERT = "family_reranker_predictions_v4_species.tsv"

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
    header = next(f)
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

print("LLM HURTS: BERT correct but LLM wrong")
print("case_id\tmention\tgold\tbert_pred\tbert_symbol\tbert_taxid\tfocus_taxid\tllm_pred")

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
        continue

    i, b = found
    used.add(i)

    pred = llm_pred.get(g["case_id"])
    if not pred:
        continue

    llm_correct = pred in g["gold"]

    if b["bert_correct"] and not llm_correct:
        print(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                g["case_id"],
                g["mention"],
                "|".join(sorted(g["gold"])),
                b["bert_pred"],
                b["bert_symbol"],
                b["bert_taxid"],
                b["focus_taxid"],
                pred,
            )
        )
