# -*- coding: utf-8 -*-

import re
from collections import Counter, defaultdict

import pandas as pd

GOLD = "bioelqa_dev_symbol_gold.tsv"
SYMBOL_PRED = "bioelqa_dev_symbol_predictions_qwen14.txt"
NCBI_KB = "ncbi_symbol_synonym_taxid_kb.tsv"
DEV_MENTIONS = "bioelqa_dev_mentions.tsv"

PRIOR_TAX = ["9606", "10090", "10116"]


def normalize_symbol(sym):
    sym = str(sym).strip()
    sym = sym.replace("Gene Symbol:", "").strip()
    sym = sym.replace("Symbol:", "").strip()
    sym = sym.strip("`'\".,;:()[]{}")
    return sym.upper()


def detect_species_from_context(text):
    t = str(text).lower()

    mouse_terms = [
        "mouse", "mice", "murine", "mus musculus",
        "c57bl", "balb/c", "129/sv", "nod mice"
    ]
    rat_terms = [
        "rat", "rats", "rattus", "sprague-dawley",
        "wistar", "wk y", "sh r", "f344"
    ]
    human_terms = [
        "human", "humans", "patient", "patients",
        "clinical", "homo sapiens", "men", "women"
    ]

    if any(x in t for x in mouse_terms):
        return "10090", "rule_mouse"
    if any(x in t for x in rat_terms):
        return "10116", "rule_rat"
    if any(x in t for x in human_terms):
        return "9606", "rule_human"

    return "", "rule_unclear"


def read_gold(path):
    gold = {}
    case2doc = {}
    case2mention = {}

    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                parts = line.rstrip("\n").split()
            if len(parts) < 4:
                continue

            case_id, doc_id, mention, gids = parts[:4]
            gold[case_id] = set(str(gids).split("|"))
            case2doc[case_id] = doc_id
            case2mention[case_id] = mention

    return gold, case2doc, case2mention


def read_predictions(path):
    symbol_pred = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            m_case = re.search(r"(dev_symbol_case_\d+)", line)
            m_sym = re.search(r"Symbol:\s*(.+)", line)

            if not m_case:
                continue

            sym = m_sym.group(1).strip() if m_sym else "NONE"
            symbol_pred[m_case.group(1)] = normalize_symbol(sym)

    return symbol_pred


def read_contexts(path):
    df = pd.read_csv(path, sep="\t")
    case2context = {}

    for i, row in df.iterrows():
        cid = f"dev_symbol_case_{i + 1}"
        case2context[cid] = row["context"]

    return case2context


def load_ncbi_kb(path):
    term_taxid2gid = {}
    term2all = defaultdict(list)

    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            term, tax_id, gid, count = parts[:4]
            key = normalize_symbol(term)

            try:
                count = int(count)
            except Exception:
                count = 0

            term_taxid2gid[(key, tax_id)] = gid
            term2all[key].append((tax_id, gid, count))

    return term_taxid2gid, term2all


def map_symbol_to_gid_with_focus(sym, focus_taxid, term_taxid2gid, term2all):
    if not sym or sym == "NONE":
        return "NONE", "none"

    if focus_taxid and (sym, focus_taxid) in term_taxid2gid:
        return term_taxid2gid[(sym, focus_taxid)], "focus_species"

    if sym in term2all:
        candidates = term2all[sym]
        unique_gids = sorted(set(gid for taxid, gid, count in candidates))

        if len(unique_gids) == 1:
            return unique_gids[0], "unique_global"

        for tax in PRIOR_TAX:
            tax_candidates = [
                (gid, count)
                for taxid, gid, count in candidates
                if taxid == tax
            ]
            if tax_candidates:
                tax_candidates = sorted(tax_candidates, key=lambda x: -x[1])
                return tax_candidates[0][0], f"prior_tax_{tax}"

        candidates = sorted(candidates, key=lambda x: -x[2])
        return candidates[0][1], "count_fallback"

    return "NONE", "unmapped"


gold, case2doc, case2mention = read_gold(GOLD)
symbol_pred = read_predictions(SYMBOL_PRED)
case2context = read_contexts(DEV_MENTIONS)

print("loaded gold:", len(gold))
print("loaded predictions:", len(symbol_pred))
print("loaded contexts:", len(case2context))

print("loading NCBI KB...")
term_taxid2gid, term2all = load_ncbi_kb(NCBI_KB)
print("loaded KB terms:", len(term2all))

total = len(gold)
done = 0
mapped = 0
correct = 0

mode_counter = Counter()
species_counter = Counter()
unmapped = Counter()
wrong = Counter()

for cid, gids in gold.items():
    if cid not in symbol_pred:
        continue

    done += 1

    sym = symbol_pred[cid]
    context = case2context.get(cid, "")
    focus_taxid, sp_mode = detect_species_from_context(context)

    gid, map_mode = map_symbol_to_gid_with_focus(
        sym,
        focus_taxid,
        term_taxid2gid,
        term2all
    )

    mode = sp_mode + "+" + map_mode

    mode_counter[mode] += 1
    species_counter[sp_mode] += 1

    if gid != "NONE":
        mapped += 1
    else:
        unmapped[sym] += 1

    if gid in gids:
        correct += 1
    else:
        mention = case2mention.get(cid, "")
        wrong[(mention, sym, gid, "|".join(sorted(gids)), focus_taxid, mode)] += 1

print("\n=== DEV Symbol → GeneID Evaluation with Context Species Rules ===")
print("total gold:", total)
print("evaluated:", done)
print("symbol mapped:", mapped)
print("correct:", correct)
print("accuracy on evaluated:", correct / done if done else 0)
print("mapping coverage:", mapped / done if done else 0)

print("\nDetected species modes:")
for k, v in species_counter.most_common():
    print(k, v)

print("\nMapping modes:")
for k, v in mode_counter.most_common():
    print(k, v)

print("\nTop unmapped symbols:")
for sym, cnt in unmapped.most_common(30):
    print(cnt, sym)

print("\nTop wrong mapped symbols:")
for (mention, sym, gid, gold_gids, focus_taxid, mode), cnt in wrong.most_common(30):
    print(
        cnt,
        "mention=", mention,
        "pred_sym=", sym,
        "mapped_gid=", gid,
        "gold=", gold_gids,
        "focus_taxid=", focus_taxid,
        "mode=", mode
    )
