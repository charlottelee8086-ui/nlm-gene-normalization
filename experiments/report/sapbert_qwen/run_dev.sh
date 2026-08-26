#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CANDIDATES="$ROOT_DIR/data/processed/dev_sapbert_top20.tsv"
PROMPTS="$ROOT_DIR/data/processed/dev_sapbert_top20_prompts.jsonl"
PREDICTIONS="$ROOT_DIR/data/processed/dev_sapbert_top20_predictions.tsv"

python "$ROOT_DIR/src/prompts/build_candidate_prompts.py" \
    --candidates "$CANDIDATES" \
    --context-column context \
    --template sapbert \
    --prompts-output "$PROMPTS"

python "$ROOT_DIR/src/inference/run_qwen_selector.py" \
    --input "$PROMPTS" \
    --output "$PREDICTIONS"

python "$ROOT_DIR/src/evaluation/evaluate_candidates.py" \
    --candidates "$CANDIDATES" \
    --predictions "$PREDICTIONS"
