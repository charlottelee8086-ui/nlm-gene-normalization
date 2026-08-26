from pathlib import Path
import re

IN = Path("pubmedbert_ner_test_predictions.tsv")
OUT = Path("pubmedbert_ner_cleaned.tsv")

# GNormPlus linker 经常不能处理的 family/pathway/group mentions
DROP_TERMS = {
    "cytokines", "cytokine", "chemokines", "chemokine",
    "mapk", "wnt", "nf-kappab", "nf-κb",
    "erk1/2", "mek1/2", "gpcr", "gpcrs",
    "mcm2-7", "orc", "sac", "sac proteins",
    "kinase", "kinases",
    "receptor", "receptors",
}

# 删除一些明显不是 gene mention 的普通词
BAD_TERMS = {
    "transcription", "adrenaline", "histone", "complex iii",
    "actin filaments", "corticosterone",
}

# 如果 mention 以这些后缀结尾，尝试去掉
TRAILING_WORDS = [
    " proteins", " protein", " genes", " gene",
    " receptors", " receptor",
    " pathway", " signaling",
    " family", " complex",
]

def normalize_key(s):
    s = s.lower()
    s = s.replace("-", "")
    return re.sub(r"\s+", " ", s).strip()

def trim_mention(mention, start, end):
    m = mention

    changed = True
    while changed:
        changed = False
        for suffix in TRAILING_WORDS:
            if m.lower().endswith(suffix):
                m = m[: -len(suffix)]
                end = start + len(m)
                changed = True

    return m.strip(), start, end

kept = 0
dropped = 0
trimmed = 0

with open(IN, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue

        pmid, start, end, mention, etype = parts[:5]
        start = int(start)
        end = int(end)

        key = normalize_key(mention)

        if key in DROP_TERMS or key in BAD_TERMS:
            dropped += 1
            continue

        new_mention, new_start, new_end = trim_mention(mention, start, end)

        if new_mention != mention:
            trimmed += 1

        if not new_mention:
            dropped += 1
            continue

        out.write(f"{pmid}\t{new_start}\t{new_end}\t{new_mention}\tGene\n")
        kept += 1

print("kept:", kept)
print("dropped:", dropped)
print("trimmed:", trimmed)
print("saved:", OUT)
