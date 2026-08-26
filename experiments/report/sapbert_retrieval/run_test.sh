#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python "$ROOT_DIR/src/retrieval/sapbert.py" \
    --preset report-test \
    --mentions "$ROOT_DIR/data/processed/test_mentions.tsv" \
    --dictionary "$ROOT_DIR/data/processed/ncbi_symbol_synonym_taxid_kb.tsv" \
    --output "$ROOT_DIR/data/processed/test_sapbert_top20.tsv" \
    --k 20
