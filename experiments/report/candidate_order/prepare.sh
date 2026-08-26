#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

ORIGINAL="$ROOT_DIR/data/processed/dev_dictionary_top10_full_abstract.tsv"

if [ ! -f "$ORIGINAL" ]; then
    echo "Missing original candidate file:"
    echo "$ORIGINAL"
    echo
    echo "Run the dictionary full-abstract preparation first."
    exit 1
fi

for SEED in 1 2 3 4 5
do
    SHUFFLED="$ROOT_DIR/data/processed/candidate_order_seed${SEED}.tsv"
    MAP="$ROOT_DIR/data/processed/candidate_order_seed${SEED}_map.tsv"
    PROMPTS="$ROOT_DIR/data/processed/candidate_order_seed${SEED}_prompts.jsonl"

    python "$ROOT_DIR/src/data/shuffle_candidates.py" \
        --input "$ORIGINAL" \
        --output "$SHUFFLED" \
        --map-output "$MAP" \
        --seed "$SEED"

    python "$ROOT_DIR/src/prompts/build_candidate_prompts.py" \
        --candidates "$SHUFFLED" \
        --context-column ctx_abstract \
        --template dictionary \
        --prompts-output "$PROMPTS"
done

echo
echo "Prepared five candidate-order permutations."
