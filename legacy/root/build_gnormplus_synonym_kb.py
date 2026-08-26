from pathlib import Path
import re

DICT = Path.home() / "nlm_gene_repro/GNorm2/Dictionary/PT_Gene.txt"
OUT = Path("gnormplus_synonym_kb.tsv")

def parse_candidate_field(field):
    """
    Example:
    9606:9466-3562,9244-3489|10090:218624-51395
    Return gene IDs only.
    """
    gids = []

    for block in field.split("|"):
        if ":" not in block:
            continue

        taxid, rest = block.split(":", 1)

        for item in rest.split(","):
            item = item.strip().lstrip("*")
            if not item:
                continue

            if "-" in item:
                gid = item.split("-")[0]
            else:
                gid = item

            if gid.isdigit():
                gids.append(gid)

    return gids

pairs = set()
bad = 0

with open(DICT, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 3:
            continue

        synonym = parts[1].strip()
        cand_field = parts[2].strip()

        if not synonym:
            continue

        gids = parse_candidate_field(cand_field)

        if not gids:
            bad += 1
            continue

        for gid in gids:
            pairs.add((synonym, gid))

with open(OUT, "w", encoding="utf-8") as out:
    for synonym, gid in sorted(pairs):
        out.write(f"{synonym}\t{gid}\n")

print("saved:", OUT)
print("pairs:", len(pairs))
print("bad:", bad)
