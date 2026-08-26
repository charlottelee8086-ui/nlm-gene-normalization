import json
import re
from pathlib import Path

ALIAS_MAP = Path("ncbi_gene_alias_map.tsv")
TRAIN_IN = Path("family_rerank_train.jsonl")
TEST_IN = Path("family_rerank_candidates.jsonl")

TRAIN_OUT = Path("family_pairwise_train_v4.jsonl")
TEST_OUT = Path("family_pairwise_test_v4.jsonl")


def norm(s):
    s = s.lower()
    s = s.replace("κ", "kappa")
    s = s.replace("β", "beta")
    s = s.replace("α", "alpha")
    s = s.replace("γ", "gamma")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


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


def lexical_features(mention, context, symbol, aliases):
    m = norm(mention)
    c = norm(context)
    terms = [symbol] + [x.strip() for x in aliases.split(";") if x.strip()]

    exact_mention_match = False
    appears_in_context = False
    overlap_terms = []

    for term in terms:
        t = norm(term)
        if not t or len(t) < 2:
            continue
        if m == t:
            exact_mention_match = True
            overlap_terms.append(term)
        if t in c:
            appears_in_context = True
            overlap_terms.append(term)
        if t in m or m in t:
            overlap_terms.append(term)

    overlap_terms = sorted(set(overlap_terms))[:10]

    return {
        "exact_mention_match": exact_mention_match,
        "appears_in_context": appears_in_context,
        "overlap_terms": "; ".join(overlap_terms) if overlap_terms else "NONE",
    }


gid2info = load_alias_map()


def convert(inp, outp, filter_gold_in=True):
    n_mentions = 0
    n_pairs = 0
    n_pos = 0
    n_neg = 0

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
                info = gid2info.get(gid, {
                    "tax_id": "UNKNOWN",
                    "symbol": "UNKNOWN",
                    "description": "UNKNOWN",
                    "aliases": "UNKNOWN",
                })

                lex = lexical_features(
                    ex["mention"],
                    ex["context"],
                    info["symbol"],
                    info["aliases"],
                )

                pair = {
                    "pmid": ex["pmid"],
                    "mention": ex["mention"],
                    "context": ex["context"],
                    "candidate_gene_id": gid,
                    "candidate_tax_id": info["tax_id"],
                    "candidate_symbol": info["symbol"],
                    "candidate_description": info["description"],
                    "candidate_aliases": info["aliases"],
                    "exact_mention_match": lex["exact_mention_match"],
                    "appears_in_context": lex["appears_in_context"],
                    "overlap_terms": lex["overlap_terms"],
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


convert(TRAIN_IN, TRAIN_OUT, filter_gold_in=True)
convert(TEST_IN, TEST_OUT, filter_gold_in=False)

