from pathlib import Path
from collections import Counter

ner_path = Path("pubmedbert_ner_test_predictions.tsv")
norm_path = Path.home() / "nlm_gene_repro/GNorm2/hybrid_norm_output/nlm_gene_test.PubTator"

ner = set()
ner_full = []

with open(ner_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        pmid, start, end, mention, etype = parts[:5]
        if etype != "Gene":
            continue
        key = (pmid, int(start), int(end), mention)
        ner.add((pmid, int(start), int(end)))
        ner_full.append(key)

norm = set()

with open(norm_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        pmid, start, end, mention, etype, gid = parts[:6]
        if etype != "Gene":
            continue
        norm.add((pmid, int(start), int(end)))

dropped = []

for pmid, start, end, mention in ner_full:
    if (pmid, start, end) not in norm:
        dropped.append((pmid, start, end, mention))

print("NER predictions:", len(ner_full))
print("Normalized predictions:", len(norm))
print("Dropped by linker:", len(dropped))
print()

print("=== Top dropped mentions ===")
for mention, count in Counter(x[3] for x in dropped).most_common(80):
    print(count, mention)
