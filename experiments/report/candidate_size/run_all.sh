#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

for K in 10 15 20
do
    echo
    echo "========================================"
    echo "Dictionary candidates: K=$K"
    echo "========================================"

    "$ROOT_DIR/experiments/report/dictionary_full_abstract/run.sh" "$K"
done
