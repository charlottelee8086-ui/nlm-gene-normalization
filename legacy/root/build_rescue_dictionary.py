IN = "top_single_gene_missing.tsv"
OUT = "rescue_dictionary.tsv"

MIN_FREQ = 2

MIN_RATIO = 0.8

with open(IN, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    header = next(f)
    out.write("mention\tgene_id\tfreq\tratio\n")

    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        mention, freq, top_gid, top_count, all_gids = parts
        freq = int(freq)
        top_count = int(top_count)

        ratio = top_count / freq

        if freq >= MIN_FREQ and ratio >= MIN_RATIO and top_gid:
            out.write(f"{mention}\t{top_gid}\t{freq}\t{ratio:.3f}\n")

print("saved:", OUT)
