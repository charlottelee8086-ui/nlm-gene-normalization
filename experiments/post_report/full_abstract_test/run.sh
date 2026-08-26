#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

K=${1:-10}

case "$K" in
    10|20)
        ;;
    *)
        echo "This experiment was run with K = 10 or K = 20."
        exit 1
        ;;
esac

MENTIONS="$ROOT_DIR/data/processed/test_mentions_with_contexts.tsv"
DICTIONARY="$ROOT_DIR/data/processed/ncbi_symbol_synonym_taxid_kb.tsv"

CANDIDATES="$ROOT_DIR/data/processed/test_dictionary_top${K}_full_abstract.tsv"
PROMPTS="$ROOT_DIR/data/processed/test_dictionary_top${K}_full_abstract_prompts.jsonl"
PREDICTIONS="$ROOT_DIR/data/processed/test_dictionary_top${K}_full_abstract_predictions.tsv"
SKIPPED="$ROOT_DIR/data/processed/test_dictionary_top${K}_full_abstract_skipped.tsv"

if [ ! -f "$MENTIONS" ]; then
    echo "Missing input file:"
    echo "$MENTIONS"
    echo
    echo "The input should contain the official test mentions and the ctx_abstract column."
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
    --case-prefix test_mcqa_case

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
