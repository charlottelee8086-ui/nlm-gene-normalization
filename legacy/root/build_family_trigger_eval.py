import csv

PRED = "family_reranker_predictions_v5_llm_species.tsv"

TRIGGER = {
    "MAPK",
    "MAPK (p38)",
    "mitogen-activated protein kinase",
    "mitogen-activated protein kinases",
    "mitogen activated protein kinase",
    "NF-kappaB",
    "NF-κB",
    "nuclear factor (NF)-κB",
    "H3",
    "(H3",
    "histone",
    "WNT",
    "Wnt",
    "Wnts",
    "ERK1/2",
    "chemokines",
    "chemokine",
}

total = 0
trigger_total = 0
trigger_wrong_oracle = 0
trigger_correct = 0

non_trigger_total = 0
non_trigger_wrong_oracle = 0

with open(PRED, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        total += 1

        mention = row["mention"]
        correct = row["correct"] == "1"
        oracle = row["oracle"] == "1"

        if mention in TRIGGER:
            trigger_total += 1
            trigger_correct += int(correct)

            if (not correct) and oracle:
                trigger_wrong_oracle += 1
        else:
            non_trigger_total += 1

            if (not correct) and oracle:
                non_trigger_wrong_oracle += 1

print("total:", total)
print("trigger_total:", trigger_total)
print("trigger_correct:", trigger_correct)
print("trigger_wrong_oracle:", trigger_wrong_oracle)
print("non_trigger_total:", non_trigger_total)
print("non_trigger_wrong_oracle:", non_trigger_wrong_oracle)
print("trigger_wrong_oracle_rate:", trigger_wrong_oracle / trigger_total if trigger_total else 0)
print("non_trigger_wrong_oracle_rate:", non_trigger_wrong_oracle / non_trigger_total if non_trigger_total else 0)
