import json
from pathlib import Path

TRAIN_IN = Path("family_rerank_train.jsonl")
TEST_IN = Path("family_rerank_candidates.jsonl")

TRAIN_OUT = Path("family_pairwise_train.jsonl")
TEST_OUT = Path("family_pairwise_test.jsonl")


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

                pair = {
                    "pmid": ex["pmid"],
                    "mention": ex["mention"],
                    "context": ex["context"],
                    "candidate_gene_id": gid,
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

