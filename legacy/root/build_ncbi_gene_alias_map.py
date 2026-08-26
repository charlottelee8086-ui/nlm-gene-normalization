from pathlib import Path

IN = Path("gene_info")
OUT = Path("ncbi_gene_alias_map.tsv")

with open(IN, encoding="utf-8", errors="ignore") as f, open(OUT, "w", encoding="utf-8") as out:
    out.write("gene_id\ttax_id\tsymbol\tdescription\taliases\n")

    for line in f:
        if line.startswith("#"):
            continue

        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue

        tax_id = parts[0]
        gene_id = parts[1]
        symbol = parts[2]
        synonyms = parts[4]
        description = parts[8]

        alias_list = []
        alias_list.append(symbol)

        if synonyms and synonyms != "-":
            alias_list.extend(synonyms.split("|"))

        alias_list = sorted(set(x.strip() for x in alias_list if x.strip()))

        out.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                gene_id,
                tax_id,
                symbol,
                description,
                "; ".join(alias_list),
            )
        )

print("saved:", OUT)
