import json

IN = "species_ambiguity_with_context.jsonl"
OUT = "llm_species_lookup.tsv"

with open(IN, encoding="utf-8") as f, \
     open(OUT, "w", encoding="utf-8") as out:

    out.write(
        "pmid\tmention\tllm_taxid\n"
    )

    species_map = {}

    with open("llm_species_map.tsv", encoding="utf-8") as sf:
        next(sf)

        for line in sf:
            case_id, taxid = line.rstrip().split("\t")
            species_map[case_id] = taxid

    for line in f:

        ex = json.loads(line)

        taxid = species_map.get(
            ex["case_id"],
            "unclear"
        )

        out.write(
            "{}\t{}\t{}\n".format(
                ex["pmid"],
                ex["mention"],
                taxid,
            )
        )

print("saved:", OUT)
