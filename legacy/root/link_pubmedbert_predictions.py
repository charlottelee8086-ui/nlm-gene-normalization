from pathlib import Path

pred_path = Path("pubmedbert_ner_test_predictions.tsv")
dict_path = Path("linking_dictionary.tsv")
out_path = Path("pubmedbert_linked_predictions.PubTator")

mention2id = {}

with open(dict_path, encoding="utf-8") as f:
    for line in f:
        mention, gid = line.rstrip("\n").split("\t")
        mention2id[mention] = gid

linked = 0
unlinked = 0

with open(pred_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        pmid, start, end, mention, etype = parts[:5]

        key = mention.lower()

        if key not in mention2id:
            unlinked += 1
            continue

        gid = mention2id[key]
        linked += 1

        out.write(f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{gid}\n")

print("Saved:", out_path)
print("Linked:", linked)
print("Unlinked:", unlinked)
