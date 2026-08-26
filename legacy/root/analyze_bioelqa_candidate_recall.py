from pathlib import Path
import pandas as pd
from collections import defaultdict

DEV = Path("bioelqa_dev_mentions.tsv")
KB = Path("ncbi_symbol_synonym_taxid_kb.tsv")

TOPKS = [1, 5, 10, 20]

def norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "").strip()

print("Loading dev...")
dev = pd.read_csv(DEV, sep="\t")

print("Loading KB...")
alias_to_geneids = defaultdict(list)

with open(KB, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline().rstrip("\n").split("\t")
    print("KB header:", header)

    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue

        term, tax_id, gene_id, count = parts[:4]

        nk = norm(term)
        if nk:
            alias_to_geneids[nk].append(str(gene_id))

print("Alias keys:", len(alias_to_geneids))

def candidates_for_mention(mention, max_k=20):
    m = norm(mention)
    cands = []

    # exact normalized alias match
    if m in alias_to_geneids:
        cands.extend(alias_to_geneids[m])

    # 去重，保留顺序
    seen = set()
    out = []
    for g in cands:
        if g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= max_k:
            break
    return out

total = 0
hit_at = {k: 0 for k in TOPKS}
no_candidate = 0
examples_no_candidate = []
examples_miss = []

for _, r in dev.iterrows():
    total += 1
    mention = r["mention"]
    golds = set(str(r["gold_geneid"]).split("|"))

    cands = candidates_for_mention(mention, max_k=max(TOPKS))

    if not cands:
        no_candidate += 1
        if len(examples_no_candidate) < 10:
            examples_no_candidate.append((mention, r["gold_geneid"]))
        continue

    for k in TOPKS:
        if golds.intersection(set(cands[:k])):
            hit_at[k] += 1

    if not golds.intersection(set(cands[:max(TOPKS)])) and len(examples_miss) < 10:
        examples_miss.append((mention, r["gold_geneid"], ",".join(cands[:10])))

print("\n=== Candidate Recall on DEV ===")
print("Total mentions:", total)
print("No candidate:", no_candidate, f"({no_candidate/total:.2%})")

for k in TOPKS:
    print(f"Recall@{k}: {hit_at[k]}/{total} = {hit_at[k]/total:.2%}")

print("\nExamples with no candidate:")
for x in examples_no_candidate:
    print(x)

print("\nExamples where gold not in top candidates:")
for x in examples_miss:
    print(x)
