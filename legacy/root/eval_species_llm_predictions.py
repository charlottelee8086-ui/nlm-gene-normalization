import json
import re

PRED = "species_llm_predictions.txt"
CASES = "species_ambiguity_with_context.jsonl"

# Build case info.
cases = {}

with open(CASES, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        gold_gids = set(ex["gold_gene_ids"].split("|"))

        cases[ex["case_id"]] = {
            "mention": ex["mention"],
            "gold_gene_ids": gold_gids,
            "focus_taxid": ex["focus_taxid"],
            "pred_gid": ex["pred_gid"],
            "pred_taxid": ex["pred_taxid"],
        }

# Load LLM species predictions.
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
llm_matches_focus = 0
llm_matches_gold_species_proxy = 0
potential_fix = 0

print("case_id\tmention\tgold\told_pred\told_tax\tfocus\tllm_tax\tpotential_fix")

for case_id, ex in cases.items():
    total += 1

    tax = llm_taxid.get(case_id, "missing")

    # We do not directly know gold taxid here unless we map gene_id->taxid.
    # For now, use focus_taxid as a weak proxy and estimate whether LLM could fix old species mismatch.
    if tax == ex["focus_taxid"]:
        llm_matches_focus += 1

    # Potential fix: old prediction taxid differs from LLM taxid.
    # This suggests LLM would push away from current wrong species.
    could_fix = (
        tax not in {"missing", "unclear"}
        and tax != ex["pred_taxid"]
    )

    potential_fix += int(could_fix)

    print(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            case_id,
            ex["mention"],
            "|".join(sorted(ex["gold_gene_ids"])),
            ex["pred_gid"],
            ex["pred_taxid"],
            ex["focus_taxid"],
            tax,
            int(could_fix),
        )
    )

print()
print("total:", total)
print("LLM matches GNormPlus focus_taxid:", llm_matches_focus)
print("potential species fixes:", potential_fix)
