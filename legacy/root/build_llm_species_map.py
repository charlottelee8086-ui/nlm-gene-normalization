import re

IN = "species_llm_predictions.txt"
OUT = "llm_species_map.tsv"

with open(IN, encoding="utf-8") as f, \
     open(OUT, "w", encoding="utf-8") as out:

    out.write("case_id\tllm_taxid\n")

    for line in f:
        line = line.strip()

        if not line:
            continue

        m_case = re.search(r"(species_case_\d+)", line)
        m_tax = re.search(r"TaxID:\s*(\d+|unclear)", line)

        if not m_case or not m_tax:
            continue

        out.write(
            "{}\t{}\n".format(
                m_case.group(1),
                m_tax.group(1),
            )
        )

print("saved:", OUT)
