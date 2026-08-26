# -*- coding: utf-8 -*-

import re

PRED = "direct_llm_symbol_predictions_qwen14.txt"

kb = set()

for path in [
    "train_synonym_kb.tsv",
    "gnormplus_synonym_kb_filtered.tsv"
]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue

            kb.add(parts[0].strip().upper())

unmapped = []

with open(PRED, encoding="utf-8") as f:
    for line in f:

        m_case = re.search(r"(direct_case_\d+)", line)
        m_sym = re.search(r"Symbol:\s*(.+)", line)

        if not m_case:
            continue

        sym = m_sym.group(1).strip() if m_sym else "NONE"

        if sym.upper() not in kb:
            unmapped.append(sym)

print("unmapped:", len(unmapped))

from collections import Counter

for sym, cnt in Counter(unmapped).most_common(100):
    print(cnt, sym)
