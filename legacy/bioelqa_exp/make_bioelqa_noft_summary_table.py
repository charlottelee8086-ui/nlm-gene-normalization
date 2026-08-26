# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


ROWS = [
    {
        "file": "bioelqa_dev_t5_noft_top5_common_species_summary.tsv",
        "dataset": "dev",
        "candidate_source": "SapBERT",
        "top_k": 5,
        "prompt": "BioELQA compact",
        "selector_model": "T5-base",
        "fine_tuning": "No",
        "setting": "BioELQA-style T5-base noFT",
    },
    {
        "file": "bioelqa_test_t5_noft_top5_summary.tsv",
        "dataset": "test",
        "candidate_source": "SapBERT",
        "top_k": 5,
        "prompt": "BioELQA compact",
        "selector_model": "T5-base",
        "fine_tuning": "No",
        "setting": "BioELQA-style T5-base noFT",
    },
    {
        "file": "bioelqa_dev_qwen25_7b_bioelqa_top5_summary.tsv",
        "dataset": "dev",
        "candidate_source": "SapBERT",
        "top_k": 5,
        "prompt": "BioELQA compact",
        "selector_model": "Qwen2.5-14B-Instruct",
        "fine_tuning": "No",
        "setting": "BioELQA-style Qwen noFT",
    },
    {
        "file": "bioelqa_test_qwen25_7b_bioelqa_top5_summary.tsv",
        "dataset": "test",
        "candidate_source": "SapBERT",
        "top_k": 5,
        "prompt": "BioELQA compact",
        "selector_model": "Qwen2.5-14B-Instruct",
        "fine_tuning": "No",
        "setting": "BioELQA-style Qwen noFT",
    },
]


out_rows = []

for meta in ROWS:
    path = Path(meta["file"])

    if not path.exists():
        print(f"Missing, skip: {path}")
        continue

    df = pd.read_csv(path, sep="\t")

    if len(df) == 0:
        print(f"Empty, skip: {path}")
        continue

    r = df.iloc[0].to_dict()

    total = int(r["total"])
    gold_in_candidates = int(r["gold_in_candidates"])
    correct = int(r["correct"])

    candidate_recall = float(r["candidate_recall_at_5"])
    final_accuracy = float(r["final_accuracy"])
    selector_acc = float(r["selector_accuracy_given_gold_in_candidates"])

    out_rows.append({
        "setting": meta["setting"],
        "dataset": meta["dataset"],
        "candidate_source": meta["candidate_source"],
        "top_k": meta["top_k"],
        "prompt": meta["prompt"],
        "selector_model": meta["selector_model"],
        "fine_tuning": meta["fine_tuning"],
        "total": total,
        "gold_in_top5": gold_in_candidates,
        "candidate_recall_at_5_pct": round(candidate_recall * 100, 2),
        "correct": correct,
        "final_accuracy_pct": round(final_accuracy * 100, 2),
        "selector_accuracy_given_gold_in_top5_pct": round(selector_acc * 100, 2),
        "valid_predictions": int(r.get("valid_predictions", 0)),
        "missing_predictions": int(r.get("missing_predictions", 0)),
        "invalid_predictions": int(r.get("invalid_predictions", 0)),
    })


out = pd.DataFrame(out_rows)

cols = [
    "setting",
    "dataset",
    "candidate_source",
    "top_k",
    "prompt",
    "selector_model",
    "fine_tuning",
    "total",
    "gold_in_top5",
    "candidate_recall_at_5_pct",
    "correct",
    "final_accuracy_pct",
    "selector_accuracy_given_gold_in_top5_pct",
    "valid_predictions",
    "missing_predictions",
    "invalid_predictions",
]

out = out[cols]

out.to_csv("bioelqa_noft_dev_test_model_comparison.tsv", sep="\t", index=False)

print(out.to_string(index=False))
print("\nSaved: bioelqa_noft_dev_test_model_comparison.tsv")
