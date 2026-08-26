from collections import defaultdict, Counter

FILE = "top_single_gene_missing.tsv"

# FamPlex-covered mentions from your analysis
FAMPLEX_MENTIONS = {
    "MAPK",
    "HIF-1α",
    "ERK1/2",
    "NF-κB",
    "NF-kappaB",
    "WNT",
    "AKT",
    "STAT",
    "CCL2",
    "CXCL9",
    "CXCL10",
    "CD44",
    "CD14",
    "MEK1/2",
    "H3",
    "histone",
    "chemokines",
    "chemokine",
}

print("Mention\tFreq\tTopGene\tTopCount\tRatio")

with open(FILE, encoding="utf-8") as f:
    next(f)

    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 5:
            continue

        mention, freq, top_gid, top_count, all_gids = parts

        if mention not in FAMPLEX_MENTIONS:
            continue

        freq = int(freq)
        top_count = int(top_count)

        ratio = top_count / freq

        print(
            "{}\t{}\t{}\t{}\t{:.3f}".format(
                mention,
                freq,
                top_gid,
                top_count,
                ratio
            )
        )


