from pathlib import Path
import pandas as pd
from collections import defaultdict

DEV = Path("bioelqa_dev_mentions.tsv")
KB = Path("ncbi_symbol_synonym_taxid_kb.tsv")

TOPKS = [1, 5, 10, 20]
PRIOR_TAX = {"9606": 0, "10090": 1, "10116": 2}

def norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "").strip()

print("Loading dev...")
dev = pd.read_csv(DEV, sep="\t")

print("Loading KB...")
alias_to_items = defaultdict(list)

with open(KB, "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        term, tax_id, gene_id, count = parts[:4]
        nk = norm(term)
        if not nk:
            continue
        try:
            count = int(count)
        except Exception:
            count = 0
        alias_to_items[nk].append((str(gene_id), str(tax_id), count))

print("Alias keys:", len(alias_to_items))

def candidates_for_mention(mention, max_k=20):
    m = norm(mention)
    items = alias_to_items.get(m, [])

    # species-aware + count-aware sorting
    items = sorted(
        items,
        key=lambda x: (
            PRIOR_TAX.get(x[1], 99),   # human/mouse/rat first
            -x[2],                     # higher count first
            x[0]
        )
    )

    seen = set()
    out = []
    for gene_id, tax_id, count in items:
        if gene_id in seen:
            continue
        seen.add(gene_id)
        out.append(gene_id)
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

print("\n=== Species-aware Candidate Recall on DEV ===")
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
