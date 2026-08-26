#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SPLIT=${1:-dev}

CANDIDATES="$ROOT_DIR/data/processed/${SPLIT}_bioelqa_top5_candidates.tsv"
PROMPTS="$ROOT_DIR/data/processed/${SPLIT}_bioelqa_prompts.jsonl"
GOLD="$ROOT_DIR/data/processed/${SPLIT}_bioelqa_gold.tsv"
PREDICTIONS="$ROOT_DIR/data/processed/${SPLIT}_bioelqa_t5_predictions.tsv"
SUMMARY="$ROOT_DIR/results/report/bioelqa_${SPLIT}_t5.csv"

if [ ! -f "$CANDIDATES" ]; then
    echo "Missing candidate file:"
    echo "$CANDIDATES"
    exit 1
fi

python "$ROOT_DIR/src/prompts/build_bioelqa_prompts.py" \
    --candidates "$CANDIDATES" \
    --prompts-output "$PROMPTS" \
    --gold-output "$GOLD" \
    --max-options 5 \
    --option-format term_species \
    --case-prefix "bioelqa_${SPLIT}"

python "$ROOT_DIR/src/inference/score_t5_letters.py" \
    --input "$PROMPTS" \
    --output "$PREDICTIONS" \
    --model t5-base

python "$ROOT_DIR/src/evaluation/evaluate_letter_selection.py" \
    --gold "$GOLD" \
    --predictions "$PREDICTIONS" \
    --summary-output "$SUMMARY" \
    --label "BioELQA-style T5-base noFT ${SPLIT}"
