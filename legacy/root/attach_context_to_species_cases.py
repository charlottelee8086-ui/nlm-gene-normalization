import json

SPECIES = "species_ambiguity_prompts.jsonl"
FAMILY = "family_rerank_candidates.jsonl"
OUT = "species_ambiguity_with_context.jsonl"

# build lookup
lookup = {}

with open(FAMILY, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        key = (
            ex["pmid"],
            ex["mention"],
            "|".join(sorted(ex["gold_gene_ids"]))
        )

        lookup[key] = ex["context"]

count = 0

with open(SPECIES, encoding="utf-8") as f, \
     open(OUT, "w", encoding="utf-8") as out:

    for line in f:

        ex = json.loads(line)

        key = (
            ex["pmid"],
            ex["mention"],
            "|".join(sorted(ex["gold_gene_ids"].split("|")))
        )

        ex["context"] = lookup.get(key, "")

        out.write(
            json.dumps(ex, ensure_ascii=False)
            + "\n"
        )

        count += 1

print("saved:", OUT)
print("cases:", count)
