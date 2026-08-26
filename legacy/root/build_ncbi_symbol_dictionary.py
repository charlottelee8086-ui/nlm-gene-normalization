# -*- coding: utf-8 -*-

from collections import defaultdict, Counter

IN = "gene_info"
OUT = "ncbi_symbol_synonym_kb.tsv"

symbol2gid = defaultdict(Counter)

with open(IN, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        tax_id = parts[0]
        gene_id = parts[1]
        symbol = parts[2]
        synonyms = parts[4]

        terms = []

        if symbol and symbol != "-":
            terms.append(symbol)

        if synonyms and synonyms != "-":
            terms.extend(synonyms.split("|"))

        for term in terms:
            term = term.strip()
            if not term:
                continue

            key = term.upper()
            symbol2gid[key][gene_id] += 1

with open(OUT, "w", encoding="utf-8") as out:
    out.write("term\tgene_id\tcount\n")

    for term, counter in sorted(symbol2gid.items()):
        gid, count = counter.most_common(1)[0]
        out.write(f"{term}\t{gid}\t{count}\n")

print("terms:", len(symbol2gid))
print("saved:", OUT)
