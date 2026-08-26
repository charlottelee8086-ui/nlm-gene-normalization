#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
K=${1:-10}

python "$ROOT_DIR/src/retrieval/dictionary.py" \
    --mentions "$ROOT_DIR/data/processed/dev_mentions.tsv" \
    --dictionary "$ROOT_DIR/data/processed/ncbi_symbol_synonym_taxid_kb.tsv" \
    --output "$ROOT_DIR/data/processed/dev_dictionary_top${K}.tsv" \
    --k "$K" \
    --case-prefix dev_mcqa_case
