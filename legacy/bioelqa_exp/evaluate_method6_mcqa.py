# -*- coding: utf-8 -*-

import argparse
import re
import pandas as pd

LETTERS = list("ABCDE")


def parse_candidates(cands_raw):
    candidates = []

    if pd.isna(cands_raw):
        return candidates

    for c in str(cands_raw).split("|"):
        parts = c.split("::", 3)
        if len(parts) != 4:
            continue

        gene_id, tax_id, species, term = parts
        candidates.append({
            "gene_id": str(gene_id),
            "tax_id": str(tax_id),
            "species": str(species),
            "term": str(term),
            "raw": c,
        })

    return candidates


def normalize_letter(x):
    if pd.isna(x):
        return ""

    s = str(x).strip()

    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1).upper()

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out_all", required=True)
    parser.add_argument("--out_errors", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    gold = pd.read_csv(args.gold, sep="\t")
    pred = pd.read_csv(args.pred, sep="\t")

    # prediction file should have: case_id, letter, decoded
    pred_map = {}
    decoded_map = {}

    for _, r in pred.iterrows():
        case_id = str(r["case_id"])
        letter = normalize_letter(r.get("letter", ""))
        decoded = str(r.get("decoded", ""))
        pred_map[case_id] = letter
        decoded_map[case_id] = decoded

    rows = []

    total = len(gold)
    gold_in_candidates_count = 0
    valid_pred_count = 0
    correct_count = 0
    missing_pred_count = 0
    invalid_pred_count = 0

    for _, r in gold.iterrows():
        case_id = str(r["case_id"])
        gold_geneids = set(str(r["gold_geneid"]).split("|"))
        candidates = parse_candidates(r["candidates"])
        cand_geneids = [c["gene_id"] for c in candidates]

        gold_in_candidates = any(g in cand_geneids for g in gold_geneids)
        if gold_in_candidates:
            gold_in_candidates_count += 1

        letter = pred_map.get(case_id, "")
        decoded = decoded_map.get(case_id, "")

        if case_id not in pred_map:
            missing_pred_count += 1

        pred_geneid = ""
        pred_species = ""
        pred_term = ""
        valid_pred = False

        if letter in LETTERS:
            idx = LETTERS.index(letter)
            if idx < len(candidates):
                valid_pred = True
                pred_geneid = candidates[idx]["gene_id"]
                pred_species = candidates[idx]["species"]
                pred_term = candidates[idx]["term"]

        if valid_pred:
            valid_pred_count += 1
        else:
            invalid_pred_count += 1

        correct = pred_geneid in gold_geneids
        if correct:
            correct_count += 1

        rows.append({
            "case_id": case_id,
            "context": args.context,
            "gold_geneid": "|".join(sorted(gold_geneids)),
            "gold_in_candidates": gold_in_candidates,
            "pred_letter": letter,
            "pred_geneid": pred_geneid,
            "pred_species": pred_species,
            "pred_term": pred_term,
            "correct": correct,
            "decoded": decoded,
            "candidates": r["candidates"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_all, sep="\t", index=False)

    errors = out[out["correct"] == False]
    errors.to_csv(args.out_errors, sep="\t", index=False)

    candidate_recall = gold_in_candidates_count / total if total else 0
    final_accuracy = correct_count / total if total else 0
    valid_rate = valid_pred_count / total if total else 0
    selector_acc_given_gold = correct_count / gold_in_candidates_count if gold_in_candidates_count else 0

    summary = pd.DataFrame([{
        "context": args.context,
        "total": total,
        "gold_in_candidates": gold_in_candidates_count,
        "candidate_recall_at_5": candidate_recall,
        "valid_predictions": valid_pred_count,
        "valid_prediction_rate": valid_rate,
        "correct": correct_count,
        "final_accuracy": final_accuracy,
        "selector_accuracy_given_gold_in_candidates": selector_acc_given_gold,
        "missing_predictions": missing_pred_count,
        "invalid_predictions": invalid_pred_count,
    }])

    summary.to_csv(args.out_summary, sep="\t", index=False)

    print("=" * 80)
    print("Context:", args.context)
    print("Total:", total)
    print("Gold in candidates:", gold_in_candidates_count)
    print("Candidate recall@5:", round(candidate_recall * 100, 2), "%")
    print("Valid predictions:", valid_pred_count)
    print("Correct:", correct_count)
    print("Final accuracy:", round(final_accuracy * 100, 2), "%")
    print("Selector accuracy given gold in candidates:", round(selector_acc_given_gold * 100, 2), "%")
    print("Missing predictions:", missing_pred_count)
    print("Invalid predictions:", invalid_pred_count)
    print("Saved:", args.out_all)
    print("Saved errors:", args.out_errors)
    print("Saved summary:", args.out_summary)


if __name__ == "__main__":
    main()
