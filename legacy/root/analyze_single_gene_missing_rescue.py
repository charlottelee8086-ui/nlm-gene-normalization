from collections import Counter, defaultdict

IN = "gold_missing_normalization.tsv"
OUT = "top_single_gene_missing.tsv"

mention_counter = Counter()
mention2gid_counter = defaultdict(Counter)

with open(IN, encoding="utf-8") as f:
    header = next(f)

    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        pmid, start, end, mention, gids = parts[:5]

        gid_list = gids.split("|")
        if len(gid_list) != 1:
            continue

        gid = gid_list[0]

        mention_counter[mention] += 1
        mention2gid_counter[mention][gid] += 1

with open(OUT, "w", encoding="utf-8") as out:
    out.write("mention\tfreq\ttop_gene_id\ttop_gene_count\tall_gene_ids\n")

    for mention, freq in mention_counter.most_common():
        gid_counter = mention2gid_counter[mention]
        top_gid, top_count = gid_counter.most_common(1)[0]

        all_gids = ";".join(
            "{}:{}".format(gid, c)
            for gid, c in gid_counter.most_common()
        )

        out.write(
            "{}\t{}\t{}\t{}\t{}\n".format(
                mention,
                freq,
                top_gid,
                top_count,
                all_gids,
            )
        )

print("single-gene missing mentions:", sum(mention_counter.values()))
print("unique single-gene missing strings:", len(mention_counter))
print("saved:", OUT)

print("\n=== Top 80 single-gene missing mentions ===\n")
for mention, freq in mention_counter.most_common(80):
    gid_counter = mention2gid_counter[mention]
    top_gid, top_count = gid_counter.most_common(1)[0]
    print(freq, mention, "=>", top_gid, "({}/{})".format(top_count, freq))
