#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "  $0 <nlm_gene_test.arrow> <gnormplus_output.PubTator>"
    exit 1
fi

python "$ROOT_DIR/src/evaluation/evaluate_gnormplus.py" \
    --gold "$1" \
    --predictions "$2" \
    --mode mention-full-test
