#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

K=${1:-10}
CONTEXT_COLUMN=${2:-context}

CANDIDATES="$ROOT_DIR/data/processed/dev_dictionary_top${K}.tsv"
PROMPTS="$ROOT_DIR/data/processed/dev_dictionary_top${K}_${CONTEXT_COLUMN}_prompts.jsonl"
PREDICTIONS="$ROOT_DIR/data/processed/dev_dictionary_top${K}_${CONTEXT_COLUMN}_predictions.tsv"
SKIPPED="$ROOT_DIR/data/processed/dev_dictionary_top${K}_${CONTEXT_COLUMN}_skipped.tsv"

python "$ROOT_DIR/src/prompts/build_candidate_prompts.py" \
    --candidates "$CANDIDATES" \
    --context-column "$CONTEXT_COLUMN" \
    --prompts-output "$PROMPTS" \
    --skipped-output "$SKIPPED"

python "$ROOT_DIR/src/inference/run_qwen_selector.py" \
    --input "$PROMPTS" \
    --output "$PREDICTIONS"

python "$ROOT_DIR/src/evaluation/evaluate_candidates.py" \
    --candidates "$CANDIDATES" \
    --predictions "$PREDICTIONS"
