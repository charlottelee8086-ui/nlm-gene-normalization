#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  $0 <context> [model]"
    echo
    echo "Contexts:"
    echo "  mention"
    echo "  sentence"
    echo "  three_sentences"
    echo "  window_500"
    echo "  abstract"
    echo "  title_abstract"
    exit 1
fi

CONTEXT="$1"
MODEL="${2:-Qwen/Qwen2.5-14B-Instruct}"

case "$CONTEXT" in
    mention|sentence|three_sentences|window_500|abstract|title_abstract)
        ;;
    *)
        echo "Unknown context: $CONTEXT"
        exit 1
        ;;
esac

PROMPTS="$ROOT_DIR/data/processed/context_${CONTEXT}_prompts.jsonl"
GOLD="$ROOT_DIR/data/processed/context_${CONTEXT}_gold.tsv"
PREDICTIONS="$ROOT_DIR/data/processed/context_${CONTEXT}_predictions.tsv"
SUMMARY="$ROOT_DIR/results/report/context_${CONTEXT}.csv"

python "$ROOT_DIR/src/inference/run_qwen_context.py" \
    --input "$PROMPTS" \
    --output "$PREDICTIONS" \
    --model "$MODEL" \
    --max-input-length 4096 \
    --max-new-tokens 16

python "$ROOT_DIR/src/evaluation/evaluate_letter_selection.py" \
    --gold "$GOLD" \
    --predictions "$PREDICTIONS" \
    --summary-output "$SUMMARY" \
    --label "Context ablation: ${CONTEXT}"
