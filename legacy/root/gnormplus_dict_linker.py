from pathlib import Path
import re

DICT_PATH = Path.home() / "nlm_gene_repro/GNorm2/Dictionary/PT_Gene.txt"
PRED_PATH = Path("pubmedbert_ner_test_predictions.tsv")
OUT_PATH = Path("pubmedbert_gnormplus_dict_linked.PubTator")

def normalize_mention(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9αβγδκ\-]+", "", s)
    s = s.replace("-", "")
    return s

def parse_candidate(field):
    # example: 9606:9466-3562,9244-3489|10090:...
    candidates = []

    for block in field.split("|"):
        if ":" not in block:
            continue

        taxid, rest = block.split(":", 1)

        for item in rest.split(","):
            item = item.strip()
            item = item.lstrip("*")

            if "-" in item:
                gene_id = item.split("-")[0]
            else:
                gene_id = item

            if gene_id.isdigit():
                candidates.append((taxid, gene_id))

    return candidates

mention2candidates = {}

with open(DICT_PATH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 3:
            continue

        surface = parts[1]
        cand_field = parts[2]

        key = normalize_mention(surface)

        if not key:
            continue

        cands = parse_candidate(cand_field)

        if cands:
            mention2candidates.setdefault(key, []).extend(cands)

print("Dictionary entries:", len(mention2candidates))

linked = 0
unlinked = 0

with open(PRED_PATH, encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:

    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 5:
            continue

        pmid, start, end, mention, etype = parts[:5]

        if etype != "Gene":
            continue

        key = normalize_mention(mention)

        if key not in mention2candidates:
            unlinked += 1
            continue

        # simple baseline: choose first candidate
        taxid, gene_id = mention2candidates[key][0]

        out.write(f"{pmid}\t{start}\t{end}\t{mention}\tGene\t{gene_id}\n")
        linked += 1

print("Linked:", linked)
print("Unlinked:", unlinked)
print("Saved:", OUT_PATH)
