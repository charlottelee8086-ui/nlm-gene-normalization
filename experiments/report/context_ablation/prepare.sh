#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "  $0 <candidate_tsv> <context_tsv>"
    echo
    echo "The candidate file must be the fixed SapBERT candidate list"
    echo "used for every context setting."
    exit 1
fi

CANDIDATES="$1"
CONTEXTS="$2"

MERGED="$ROOT_DIR/data/processed/dev_context_ablation_candidates.tsv"

python "$ROOT_DIR/src/data/attach_contexts.py" \
    --candidates "$CANDIDATES" \
    --contexts "$CONTEXTS" \
    --output "$MERGED"

for CONTEXT in \
    mention \
    sentence \
    three_sentences \
    window_500 \
    abstract \
    title_abstract
do
    python "$ROOT_DIR/src/prompts/build_context_prompts.py" \
        --input "$MERGED" \
        --context "$CONTEXT" \
        --prompts-output "$ROOT_DIR/data/processed/context_${CONTEXT}_prompts.jsonl" \
        --gold-output "$ROOT_DIR/data/processed/context_${CONTEXT}_gold.tsv" \
        --max-options 5
done

echo
echo "Context ablation inputs are ready."
