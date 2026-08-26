import json
import re

PRED = "species_llm_predictions.txt"
CASES = "species_ambiguity_with_context.jsonl"
ALIAS = "ncbi_gene_alias_map.tsv"

gid2tax = {}

with open(ALIAS, encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        gid, tax, symbol, desc, aliases = parts[:5]
        gid2tax[gid] = tax


cases = {}

with open(CASES, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        gold_gids = ex["gold_gene_ids"].split("|")
        gold_taxids = set()

        for gid in gold_gids:
            if gid in gid2tax:
                gold_taxids.add(gid2tax[gid])

        cases[ex["case_id"]] = {
            "mention": ex["mention"],
            "gold_gene_ids": gold_gids,
            "gold_taxids": gold_taxids,
            "focus_taxid": ex["focus_taxid"],
            "pred_gid": ex["pred_gid"],
            "pred_taxid": ex["pred_taxid"],
        }


llm_taxid = {}

with open(PRED, encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        m_case = re.search(r"(species_case_\d+)", line)
        m_tax = re.search(r"TaxID:\s*(\d+|unclear)", line)

        if not m_case or not m_tax:
            print("cannot parse:", line)
            continue

        llm_taxid[m_case.group(1)] = m_tax.group(1)


total = 0
old_correct_species = 0
focus_correct_species = 0
llm_correct_species = 0

llm_gain = 0
llm_hurt = 0

print("case_id\tmention\tgold_gid\tgold_tax\told_tax\tfocus\tllm_tax\told_ok\tfocus_ok\tllm_ok")

for case_id, ex in cases.items():
    total += 1

    gold_tax = ex["gold_taxids"]
    old_tax = ex["pred_taxid"]
    focus = ex["focus_taxid"]
    llm_tax = llm_taxid.get(case_id, "missing")

    old_ok = old_tax in gold_tax
    focus_ok = focus in gold_tax
    llm_ok = llm_tax in gold_tax

    old_correct_species += int(old_ok)
    focus_correct_species += int(focus_ok)
    llm_correct_species += int(llm_ok)

    if (not old_ok) and llm_ok:
        llm_gain += 1

    if old_ok and (not llm_ok):
        llm_hurt += 1

    print(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            case_id,
            ex["mention"],
            "|".join(ex["gold_gene_ids"]),
            "|".join(sorted(gold_tax)),
            old_tax,
            focus,
            llm_tax,
            int(old_ok),
            int(focus_ok),
            int(llm_ok),
        )
    )

print()
print("total:", total)
print("old pred species correct:", old_correct_species)
print("focus species correct:", focus_correct_species)
print("LLM species correct:", llm_correct_species)
print("LLM gain over old species:", llm_gain)
print("LLM hurt old correct species:", llm_hurt)
print("LLM net species gain:", llm_gain - llm_hurt)
