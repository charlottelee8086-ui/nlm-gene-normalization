#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

K=${1:-10}

MENTIONS="$ROOT_DIR/data/processed/dev_mentions_with_contexts.tsv"
DICTIONARY="$ROOT_DIR/data/processed/ncbi_symbol_synonym_taxid_kb.tsv"

CANDIDATES="$ROOT_DIR/data/processed/dev_dictionary_top${K}_full_abstract.tsv"
PROMPTS="$ROOT_DIR/data/processed/dev_dictionary_top${K}_full_abstract_prompts.jsonl"
PREDICTIONS="$ROOT_DIR/data/processed/dev_dictionary_top${K}_full_abstract_predictions.tsv"
SKIPPED="$ROOT_DIR/data/processed/dev_dictionary_top${K}_full_abstract_skipped.tsv"

if [ ! -f "$MENTIONS" ]; then
    echo "Missing input file:"
    echo "$MENTIONS"
    echo
    echo "This file should contain the development mentions and the ctx_abstract column."
    exit 1
fi

if [ ! -f "$DICTIONARY" ]; then
    echo "Missing dictionary:"
    echo "$DICTIONARY"
    exit 1
fi

python "$ROOT_DIR/src/retrieval/dictionary.py" \
    --mentions "$MENTIONS" \
    --dictionary "$DICTIONARY" \
    --output "$CANDIDATES" \
    --k "$K" \
    --case-prefix dev_mcqa_case

python "$ROOT_DIR/src/prompts/build_candidate_prompts.py" \
    --candidates "$CANDIDATES" \
    --context-column ctx_abstract \
    --template dictionary \
    --prompts-output "$PROMPTS" \
    --skipped-output "$SKIPPED"

python "$ROOT_DIR/src/inference/run_qwen_selector.py" \
    --input "$PROMPTS" \
    --output "$PREDICTIONS"

python "$ROOT_DIR/src/evaluation/evaluate_candidates.py" \
    --candidates "$CANDIDATES" \
    --predictions "$PREDICTIONS"
