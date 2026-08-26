import json
from pathlib import Path

ALIAS_MAP = Path("ncbi_gene_alias_map.tsv")
TRAIN_IN = Path("family_rerank_train.jsonl")
TEST_IN = Path("family_rerank_candidates.jsonl")

TRAIN_OUT = Path("family_pairwise_train_v3.jsonl")
TEST_OUT = Path("family_pairwise_test_v3.jsonl")


def load_alias_map():
    d = {}

    with open(ALIAS_MAP, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            gene_id, tax_id, symbol, description, aliases = parts[:5]

            d[gene_id] = {
                "tax_id": tax_id,
                "symbol": symbol,
                "description": description,
                "aliases": aliases,
            }

    return d


gid2info = load_alias_map()


def convert(inp, outp, filter_gold_in=True):
    n_mentions = 0
    n_pairs = 0
    n_pos = 0
    n_neg = 0
    missing_info = 0

    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as out:
        for line in f:
            ex = json.loads(line)
            gold = set(ex["gold_gene_ids"])
            cands = ex["candidate_gene_ids"]

            if filter_gold_in and not (gold & set(cands)):
                continue

            n_mentions += 1

            for gid in cands:
                label = 1 if gid in gold else 0
                info = gid2info.get(gid)

                if info is None:
                    missing_info += 1
                    info = {
                        "tax_id": "UNKNOWN",
                        "symbol": ex.get("candidate_name", "UNKNOWN"),
                        "description": "UNKNOWN",
                        "aliases": ex.get("candidate_name", "UNKNOWN"),
                    }

                pair = {
                    "pmid": ex["pmid"],
                    "mention": ex["mention"],
                    "context": ex["context"],
                    "candidate_gene_id": gid,
                    "candidate_tax_id": info["tax_id"],
                    "candidate_symbol": info["symbol"],
                    "candidate_description": info["description"],
                    "candidate_aliases": info["aliases"],
                    "gold_gene_ids": ex["gold_gene_ids"],
                    "label": label,
                }

                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n_pairs += 1
                if label:
                    n_pos += 1
                else:
                    n_neg += 1

    print(outp)
    print("mentions:", n_mentions)
    print("pairs:", n_pairs)
    print("positive:", n_pos)
    print("negative:", n_neg)
    print("missing info:", missing_info)


convert(TRAIN_IN, TRAIN_OUT, filter_gold_in=True)
convert(TEST_IN, TEST_OUT, filter_gold_in=False)

