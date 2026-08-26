import re

GOLD = "species_round2_gold.tsv"
PRED = "species_round2_predictions.txt"
ALIAS = "ncbi_gene_alias_map.tsv"

gid2tax = {}

with open(ALIAS, encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        gid, tax = parts[0], parts[1]
        gid2tax[gid] = tax

gold = {}

with open(GOLD, encoding="utf-8") as f:
    next(f)
    for line in f:
        case_id, pmid, mention, gold_gene_ids, pred_gid, pred_taxid, focus_taxid = line.rstrip("\n").split("\t")

        gold_taxids = set()
        for gid in gold_gene_ids.split("|"):
            if gid in gid2tax:
                gold_taxids.add(gid2tax[gid])

        gold[case_id] = {
            "mention": mention,
            "gold_gene_ids": gold_gene_ids,
            "gold_taxids": gold_taxids,
            "pred_taxid": pred_taxid,
            "focus_taxid": focus_taxid,
        }

pred = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        m_case = re.search(r"(round2_case_\d+)", line)
        m_tax = re.search(r"TaxID:\s*(\d+|unclear)", line)

        if not m_case or not m_tax:
            continue

        pred[m_case.group(1)] = m_tax.group(1)

total = 0
llm_ok = 0
focus_ok = 0
old_ok = 0

print("case_id\tmention\tgold_tax\told_tax\tfocus_tax\tllm_tax\told_ok\tfocus_ok\tllm_ok")

for case_id, ex in gold.items():
    total += 1

    llm_tax = pred.get(case_id, "missing")
    gold_taxids = ex["gold_taxids"]

    old_correct = ex["pred_taxid"] in gold_taxids
    focus_correct = ex["focus_taxid"] in gold_taxids
    llm_correct = llm_tax in gold_taxids

    old_ok += int(old_correct)
    focus_ok += int(focus_correct)
    llm_ok += int(llm_correct)

    print(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            case_id,
            ex["mention"],
            "|".join(sorted(gold_taxids)),
            ex["pred_taxid"],
            ex["focus_taxid"],
            llm_tax,
            int(old_correct),
            int(focus_correct),
            int(llm_correct),
        )
    )

print()
print("total:", total)
print("old species correct:", old_ok)
print("focus species correct:", focus_ok)
print("LLM species correct:", llm_ok)
print("LLM species acc:", llm_ok / total if total else 0)
