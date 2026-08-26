# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
import pandas as pd

DEV = Path("../bioelqa_dev_mentions.tsv")
KB = Path("../ncbi_symbol_synonym_taxid_kb.tsv")
OUT = Path("bioelqa_dev_candidates.tsv")

TOPK = 10
PRIOR_TAX = {"9606": 0, "10090": 1, "10116": 2}

def norm(s):
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "").strip()

def tax_name(taxid):
    return {
        "9606": "human",
        "10090": "mouse",
        "10116": "rat",
        "7227": "fruit fly",
        "6239": "worm",
        "3702": "Arabidopsis",
        "7955": "zebrafish",
    }.get(str(taxid), f"taxid:{taxid}")

print("Loading dev mentions...")
dev = pd.read_csv(DEV, sep="\t")

print("Loading NCBI KB...")
alias_to_items = defaultdict(list)

with open(KB, encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue

        term, tax_id, gene_id, count = parts[:4]
        key = norm(term)

        if not key:
            continue

        try:
            count = int(count)
        except Exception:
            count = 0

        alias_to_items[key].append({
            "gene_id": str(gene_id),
            "tax_id": str(tax_id),
            "term": str(term),
            "count": count,
        })

print("Alias keys:", len(alias_to_items))

def get_candidates(mention, topk=TOPK):
    key = norm(mention)
    items = alias_to_items.get(key, [])

    items = sorted(
        items,
        key=lambda x: (
            PRIOR_TAX.get(x["tax_id"], 99),
            -x["count"],
            x["gene_id"]
        )
    )

    seen = set()
    out = []

    for item in items:
        gid = item["gene_id"]
        if gid in seen:
            continue
        seen.add(gid)

        out.append(
            f'{gid}::{item["tax_id"]}::{tax_name(item["tax_id"])}::{item["term"]}'
        )

        if len(out) >= topk:
            break

    return out

rows = []
hit = 0
no_cand = 0

for i, r in dev.iterrows():
    case_id = f"dev_mcqa_case_{i+1}"
    cands = get_candidates(r["mention"], TOPK)

    golds = set(str(r["gold_geneid"]).split("|"))
    cand_gids = {c.split("::")[0] for c in cands}

    if not cands:
        no_cand += 1

    if golds.intersection(cand_gids):
        hit += 1

    rows.append({
        "case_id": case_id,
        "doc_id": r["doc_id"],
        "mention": r["mention"],
        "gold_geneid": r["gold_geneid"],
        "context": r["context"],
        "candidates": "|".join(cands),
        "gold_in_candidates": int(bool(golds.intersection(cand_gids))),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)

print("Saved:", OUT)
print("Total:", len(out))
print("No candidate:", no_cand, f"{no_cand/len(out):.2%}")
print(f"Gold in Top{TOPK}:", hit, f"{hit/len(out):.2%}")
print(out.head(3).to_string())
