import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("family_reranker_pubmedbert_v3")
TEST_FILE = Path("family_pairwise_test_v3.jsonl")
OUT = Path("family_reranker_predictions_v3.tsv")


def format_input(ex):
    return (
        "Mention: {mention}\n"
        "Context: {context}\n"
        "Candidate Gene Symbol: {symbol}\n"
        "Candidate Description: {desc}\n"
        "Candidate Aliases: {aliases}\n"
        "Candidate Gene ID: {gid}"
    ).format(
        mention=ex["mention"],
        context=ex["context"],
        symbol=ex.get("candidate_symbol", "UNKNOWN"),
        desc=ex.get("candidate_description", "UNKNOWN"),
        aliases=ex.get("candidate_aliases", "UNKNOWN"),
        gid=ex["candidate_gene_id"],
    )



device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

groups = defaultdict(list)

with open(TEST_FILE, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        key = (ex["pmid"], ex["start"] if "start" in ex else "", ex["end"] if "end" in ex else "", ex["mention"], ex["context"])
        groups[key].append(ex)

correct = 0
oracle = 0
total = 0

with open(OUT, "w", encoding="utf-8") as out:
    out.write("pmid\tmention\tpred_gid\tpred_name\tgold_gene_ids\tscore\tcorrect\toracle\n")

    for key, examples in groups.items():
        pmid = key[0]
        mention = key[3]
        gold = set(examples[0]["gold_gene_ids"])

        scores = []

        for ex in examples:
            text = format_input(ex)

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=384,
                padding="max_length",
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits[0]
                prob = torch.softmax(logits, dim=-1)[1].item()

            scores.append((
                prob,
                ex["candidate_gene_id"],
                ex.get("candidate_name", "UNKNOWN"),
            ))

        scores.sort(reverse=True)
        best_score, best_gid, best_name = scores[0]

        is_correct = best_gid in gold
        candidate_ids = {ex["candidate_gene_id"] for ex in examples}
        is_oracle = bool(gold & candidate_ids)

        total += 1
        correct += int(is_correct)
        oracle += int(is_oracle)

        out.write(
            "{}\t{}\t{}\t{}\t{}\t{:.6f}\t{}\t{}\n".format(
                pmid,
                mention,
                best_gid,
                best_name,
                "|".join(sorted(gold)),
                best_score,
                int(is_correct),
                int(is_oracle),
            )
        )

print("total mentions:", total)
print("mention-level correct:", correct)
print("mention-level accuracy:", correct / total if total else 0)
print("oracle covered:", oracle)
print("oracle accuracy:", oracle / total if total else 0)
print("saved:", OUT)
