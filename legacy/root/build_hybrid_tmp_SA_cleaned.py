from collections import defaultdict

# original tmp_SA from GNorm2
orig_path = "/mnt/beegfs/home/xli/nlm_gene_repro/GNorm2/tmp_SA/nlm_gene_test.PubTator"

# our NER predictions
pred_path = "pubmedbert_ner_cleaned.tsv"

out_path = "hybrid_tmp_SA_cleaned.PubTator"

# collect species focus per PMID
pmid2focus = defaultdict(lambda: "9606")

with open(orig_path, encoding="utf-8") as f:

    for line in f:

        if "\tSpecies\t*" not in line:
            continue

        parts = line.rstrip().split("\t")

        pmid = parts[0]

        taxid = parts[-1].replace("*", "")

        pmid2focus[pmid] = taxid

# copy title/abstract block from original
doc_lines = []

with open(orig_path, encoding="utf-8") as f:

    current = []

    for line in f:

        if line.strip() == "":
            doc_lines.append(current)
            current = []
        else:
            current.append(line)

# map pmid -> doc text block
pmid2doc = {}

for block in doc_lines:

    if not block:
        continue

    pmid = block[0].split("|")[0]

    text_lines = []

    for line in block:

        if "|t|" in line or "|a|" in line:
            text_lines.append(line)

    pmid2doc[pmid] = text_lines

# collect our predictions
pmid2preds = defaultdict(list)

with open(pred_path, encoding="utf-8") as f:

    for line in f:

        parts = line.rstrip().split("\t")

        if len(parts) < 5:
            continue

        pmid, start, end, mention, etype = parts[:5]

        if etype != "Gene":
            continue

        pmid2preds[pmid].append(
            (int(start), int(end), mention)
        )

# write hybrid tmp_SA
with open(out_path, "w", encoding="utf-8") as out:

    for pmid in pmid2doc:

        for line in pmid2doc[pmid]:
            out.write(line)

        taxid = pmid2focus[pmid]

        out.write(
            f"{pmid}\t0\t0\t{taxid}\tSpecies\t*{taxid}\n"
        )

        for start, end, mention in pmid2preds[pmid]:

            out.write(
                f"{pmid}\t{start}\t{end}\t{mention}\tGene\tFocus:{taxid}\n"
            )

        out.write("\n")

print("saved:", out_path)
