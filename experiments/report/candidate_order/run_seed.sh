#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  $0 <seed> [model]"
    echo
    echo "Seed must be 1, 2, 3, 4, or 5."
    exit 1
fi

SEED="$1"
MODEL="${2:-Qwen/Qwen2.5-14B-Instruct}"

case "$SEED" in
    1|2|3|4|5)
        ;;
    *)
        echo "Seed must be between 1 and 5."
        exit 1
        ;;
esac

ORIGINAL="$ROOT_DIR/data/processed/dev_dictionary_top10_full_abstract.tsv"
ORIGINAL_PREDICTIONS="$ROOT_DIR/data/processed/dev_dictionary_top10_full_abstract_predictions.tsv"

SHUFFLED="$ROOT_DIR/data/processed/candidate_order_seed${SEED}.tsv"
PROMPTS="$ROOT_DIR/data/processed/candidate_order_seed${SEED}_prompts.jsonl"
PREDICTIONS="$ROOT_DIR/data/processed/candidate_order_seed${SEED}_predictions.tsv"

SUMMARY="$ROOT_DIR/results/report/candidate_order_seed${SEED}_reproduced.csv"

python "$ROOT_DIR/src/inference/run_qwen_selector.py" \
    --input "$PROMPTS" \
    --output "$PREDICTIONS" \
    --model "$MODEL"

python "$ROOT_DIR/src/evaluation/evaluate_candidates.py" \
    --candidates "$SHUFFLED" \
    --predictions "$PREDICTIONS"

python "$ROOT_DIR/src/evaluation/analyze_candidate_order.py" \
    --original-candidates "$ORIGINAL" \
    --shuffled-candidates "$SHUFFLED" \
    --original-predictions "$ORIGINAL_PREDICTIONS" \
    --shuffled-predictions "$PREDICTIONS" \
    --seed "$SEED" \
    --summary-output "$SUMMARY"
