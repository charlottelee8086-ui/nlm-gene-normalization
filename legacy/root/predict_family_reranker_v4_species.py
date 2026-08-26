import json
import re
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("family_reranker_pubmedbert_v4")
TEST_FILE = Path("family_pairwise_test_v4.jsonl")
TMP_SA = Path.home() / "nlm_gene_repro/GNorm2/tmp_SA/nlm_gene_test.PubTator"
OUT = Path("family_reranker_predictions_v4_species.tsv")

SPECIES_BONUS = 0.8


def load_focus_taxid():
    pmid2taxid = {}

    with open(TMP_SA, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\tSpecies\t*" not in line:
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            pmid = parts[0]
            taxid = parts[-1].replace("*", "").strip()

            if taxid:
                pmid2taxid[pmid] = taxid

    return pmid2taxid


def norm(s):
    s = s.lower()
    s = s.replace("κ", "kappa")
    s = s.replace("β", "beta")
    s = s.replace("α", "alpha")
    s = s.replace("γ", "gamma")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def split_aliases(s):
    if not s or s == "UNKNOWN":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def lexical_bonus(ex):
    mention = norm(ex["mention"])
    context = norm(ex["context"])

    symbol = ex.get("candidate_symbol", "")
    aliases = split_aliases(ex.get("candidate_aliases", ""))

    terms = [symbol] + aliases
    bonus = 0.0

    for term in terms:
        t = norm(term)
        if not t or len(t) < 2:
            continue

        if mention == t:
            bonus += 1.0

        if t in context:
            bonus += 0.6

        if t in mention or mention in t:
            bonus += 0.3

    return min(bonus, 2.0)


def species_bonus(ex, focus_taxid):
    cand_taxid = str(ex.get("candidate_tax_id", "UNKNOWN"))

    if not focus_taxid:
        return 0.0

    if cand_taxid == focus_taxid:
        return SPECIES_BONUS

    return 0.0


def format_input(ex):
    return (
        "Mention: {mention}\n"
        "Context: {context}\n"
        "Candidate Gene Symbol: {symbol}\n"
        "Candidate Description: {desc}\n"
        "Candidate Aliases: {aliases}\n"
        "Candidate Species TaxID: {taxid}\n"
        "Exact mention match: {exact}\n"
        "Candidate appears in context: {appears}\n"
        "Overlapping terms: {overlap}\n"
        "Candidate Gene ID: {gid}"
    ).format(
        mention=ex["mention"],
        context=ex["context"],
        symbol=ex.get("candidate_symbol", "UNKNOWN"),
        desc=ex.get("candidate_description", "UNKNOWN"),
        aliases=ex.get("candidate_aliases", "UNKNOWN"),
        taxid=ex.get("candidate_tax_id", "UNKNOWN"),
        exact=str(ex.get("exact_mention_match", False)),
        appears=str(ex.get("appears_in_context", False)),
        overlap=ex.get("overlap_terms", "NONE"),
        gid=ex["candidate_gene_id"],
    )


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

pmid2taxid = load_focus_taxid()

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

groups = defaultdict(list)

with open(TEST_FILE, encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        key = (
            ex["pmid"],
            ex.get("start", ""),
            ex.get("end", ""),
            ex["mention"],
            ex["context"],
        )
        groups[key].append(ex)

correct = 0
oracle = 0
total = 0

with open(OUT, "w", encoding="utf-8") as out:
    out.write(
        "pmid\tmention\tfocus_taxid\tpred_gid\tpred_symbol\tpred_taxid\tgold_gene_ids\t"
        "neural_score\tlexical_bonus\tspecies_bonus\tfinal_score\tcorrect\toracle\n"
    )

    for key, examples in groups.items():
        pmid = key[0]
        mention = key[3]
        focus_taxid = pmid2taxid.get(pmid, "")

        gold = set(examples[0]["gold_gene_ids"])
        scores = []

        for ex in examples:
            text = format_input(ex)

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=384,
                padding="max_length",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits[0]
                neural = torch.softmax(logits, dim=-1)[1].item()

            lex = lexical_bonus(ex)
            sp = species_bonus(ex, focus_taxid)
            final = neural + lex + sp

            scores.append((
                final,
                neural,
                lex,
                sp,
                ex["candidate_gene_id"],
                ex.get("candidate_symbol", "UNKNOWN"),
                ex.get("candidate_tax_id", "UNKNOWN"),
            ))

        scores.sort(reverse=True)
        best_final, best_neural, best_lex, best_sp, best_gid, best_symbol, best_taxid = scores[0]

        candidate_ids = {ex["candidate_gene_id"] for ex in examples}
        is_correct = best_gid in gold
        is_oracle = bool(gold & candidate_ids)

        total += 1
        correct += int(is_correct)
        oracle += int(is_oracle)

        out.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.3f}\t{:.3f}\t{:.6f}\t{}\t{}\n".format(
                pmid,
                mention,
                focus_taxid,
                best_gid,
                best_symbol,
                best_taxid,
                "|".join(sorted(gold)),
                best_neural,
                best_lex,
                best_sp,
                best_final,
                int(is_correct),
                int(is_oracle),
            )
        )

print("total mentions:", total)
print("mention-level correct:", correct)
print("mention-level accuracy:", correct / total if total else 0)
print("oracle covered:", oracle)
print("oracle accuracy:", oracle / total if total else 0)
print("saved:", OUT)
