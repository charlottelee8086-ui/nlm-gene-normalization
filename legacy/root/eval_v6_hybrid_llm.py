import csv

BASE = "family_reranker_predictions_v5_llm_species.tsv"
LOOKUP = "family_llm_lookup.tsv"

TOTAL_GOLD = 2729
BASE_SYSTEM_CORRECT = 1705  # synonym rescue stage correct
# family reranker v5 contributes 85 correct on 147 family candidate cases

family_llm = {}

with open(LOOKUP, encoding="utf-8") as f:
    next(f)
    for line in f:
        pmid, mention, gold_gene_ids, llm_gid = line.rstrip("\n").split("\t")

        key = (
            pmid,
            mention,
            "|".join(sorted(gold_gene_ids.split("|"))),
        )

        family_llm[key] = llm_gid

base_correct = 0
v6_correct = 0
oracle = 0
total = 0

gain = 0
hurt = 0
used_llm = 0

with open(BASE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        total += 1

        gold = set(row["gold_gene_ids"].split("|"))

        base_ok = row["correct"] == "1"
        base_correct += int(base_ok)

        if row["oracle"] == "1":
            oracle += 1

        key = (
            row["pmid"],
            row["mention"],
            "|".join(sorted(row["gold_gene_ids"].split("|"))),
        )

        llm_gid = family_llm.get(key)

        if llm_gid:
            used_llm += 1
            final_ok = llm_gid in gold
        else:
            final_ok = base_ok

        v6_correct += int(final_ok)

        if (not base_ok) and final_ok:
            gain += 1

        if base_ok and (not final_ok):
            hurt += 1

print("family cases:", total)
print("oracle:", oracle)
print("v5 correct:", base_correct)
print("v6 correct:", v6_correct)
print("LLM used:", used_llm)
print("gain:", gain)
print("hurt:", hurt)
print("net gain:", gain - hurt)
print("v5 family accuracy:", base_correct / total if total else 0)
print("v6 family accuracy:", v6_correct / total if total else 0)

# Overall normalization estimate:
# v5 overall correct = 1705 + 85
# v6 overall correct = 1705 + v6_correct
v5_overall_correct = BASE_SYSTEM_CORRECT + base_correct
v6_overall_correct = BASE_SYSTEM_CORRECT + v6_correct

print()
print("v5 overall correct:", v5_overall_correct)
print("v5 overall accuracy:", v5_overall_correct / TOTAL_GOLD)
print("v6 overall correct:", v6_overall_correct)
print("v6 overall accuracy:", v6_overall_correct / TOTAL_GOLD)
