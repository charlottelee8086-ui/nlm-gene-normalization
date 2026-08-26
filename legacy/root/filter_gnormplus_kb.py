from pathlib import Path
import re

IN = Path("gnormplus_synonym_kb.tsv")
OUT = Path("gnormplus_synonym_kb_filtered.tsv")

def keep(s):
    if len(s) < 2 or len(s) > 60:
        return False
    if sum(c.isdigit() for c in s) > 15:
        return False
    if len(s.split()) > 8:
        return False
    # keep gene-like names
    if re.search(r"[a-zA-Z]", s):
        return True
    return False

seen = set()
kept = 0

with open(IN, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        syn, gid = line.rstrip("\n").split("\t")
        if not keep(syn):
            continue
        key = (syn, gid)
        if key in seen:
            continue
        seen.add(key)
        out.write(f"{syn}\t{gid}\n")
        kept += 1

print("kept:", kept)
print("saved:", OUT)
