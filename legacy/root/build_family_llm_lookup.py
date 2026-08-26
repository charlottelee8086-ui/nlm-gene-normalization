import re

PRED = "family_member_llm_predictions.txt"
GOLD = "family_member_llm_gold.tsv"
OUT = "family_llm_lookup.tsv"

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(family_case_\d+)", line)
        m_gid = re.search(r"GeneID:\s*(\d+)", line)

        if m_case and m_gid:
            pred[m_case.group(1)] = m_gid.group(1)

with open(GOLD, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    next(f)

    out.write("pmid\tmention\tgold_gene_ids\tllm_gene_id\n")

    for line in f:
        case_id, pmid, mention, gold_gene_ids, current_pred_gid = line.rstrip("\n").split("\t")

        llm_gid = pred.get(case_id, "")

        if not llm_gid:
            continue

        out.write(
            "{}\t{}\t{}\t{}\n".format(
                pmid,
                mention,
                gold_gene_ids,
                llm_gid,
            )
        )

print("saved:", OUT)
