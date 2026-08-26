#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python "$ROOT_DIR/src/retrieval/sapbert.py" \
    --preset report-dev \
    --mentions "$ROOT_DIR/data/processed/dev_mentions.tsv" \
    --gene-info "$ROOT_DIR/data/external/gene_info" \
    --output "$ROOT_DIR/data/processed/dev_sapbert_top20.tsv" \
    --k 20
