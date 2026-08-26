# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


KB = Path("../ncbi_symbol_synonym_taxid_kb.tsv")

# Test mentions extracted from nlm_gene-test.arrow
MENTIONS = Path("../bioelqa_dev_mentions.tsv")

# Output file
OUT = Path("bioelqa_dev_candidates_sapbert_geneinfo_top20_common_species.tsv")

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

TOPK = 20
ENCODE_BATCH_SIZE = 64
RETRIEVAL_BATCH_SIZE = 128
MAX_SYNONYMS_PER_ENTITY = 8


# Only keep common NLM-Gene species.
# This avoids encoding the full 78M synonym rows.
KEEP_TAXIDS = {
    "9606",    # human
    "10090",   # mouse
    "10116",   # rat
    "7955",    # zebrafish
    "7227",    # fruit fly
    "3702",    # arabidopsis
    "6239",    # worm
    "4932",    # yeast
}


SPECIES_NAME = {
    "9606": "human",
    "10090": "mouse",
    "10116": "rat",
    "7955": "zebrafish",
    "7227": "fruit fly",
    "3702": "arabidopsis",
    "6239": "worm",
    "4932": "yeast",
}


def safe_int(x, default=0):
    try:
        return int(float(str(x)))
    except Exception:
        return default


def load_kb(path):
    """
    Load NCBI synonym KB and aggregate synonym rows.

    Original KB:
        term    tax_id    gene_id    count

    Old version:
        one synonym row = one SapBERT entity
        78M rows -> too slow and memory-heavy

    New version:
        one GeneID + TaxID = one SapBERT entity
        entity_text = top synonyms + species
    """

    print("Loading KB from:", path)

    entity_terms = {}

    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        print("KB header:", header)

        h = [x.lower() for x in header]

        def col_index(names):
            for name in names:
                if name in h:
                    return h.index(name)
            return None

        idx_term = col_index(["term", "symbol", "name", "alias", "synonym"])
        idx_tax = col_index(["tax_id", "taxid"])
        idx_gid = col_index(["gene_id", "geneid"])
        idx_count = col_index(["count", "freq", "frequency"])

        if idx_term is None or idx_tax is None or idx_gid is None:
            raise ValueError(f"Cannot identify required columns from header: {header}")

        raw_rows = 0
        kept_rows = 0

        for line in tqdm(f, desc="reading KB"):
            raw_rows += 1

            parts = line.rstrip("\n").split("\t")

            if len(parts) <= max(idx_term, idx_tax, idx_gid):
                continue

            term = str(parts[idx_term]).strip()
            tax_id = str(parts[idx_tax]).strip()
            gene_id = str(parts[idx_gid]).strip()

            if not term or not tax_id or not gene_id:
                continue

            # Only keep selected species.
            if tax_id not in KEEP_TAXIDS:
                continue

            count = 0
            if idx_count is not None and idx_count < len(parts):
                count = safe_int(parts[idx_count], 0)

            species = SPECIES_NAME.get(tax_id, f"taxid_{tax_id}")
            key = (gene_id, tax_id, species)

            if key not in entity_terms:
                entity_terms[key] = {}

            # Keep max count for duplicate terms.
            old_count = entity_terms[key].get(term)
            if old_count is None or count > old_count:
                entity_terms[key][term] = count

            # Keep only a small number of synonyms per gene to save memory.
            if len(entity_terms[key]) > MAX_SYNONYMS_PER_ENTITY * 4:
                top_items = sorted(
                    entity_terms[key].items(),
                    key=lambda x: (-x[1], x[0])
                )[:MAX_SYNONYMS_PER_ENTITY]
                entity_terms[key] = dict(top_items)

            kept_rows += 1

    print("Raw KB rows scanned:", raw_rows)
    print("Rows kept after species filter:", kept_rows)
    print("Unique GeneID+TaxID entities:", len(entity_terms))

    rows = []

    for (gene_id, tax_id, species), term_count in entity_terms.items():
        top_terms = [
            term for term, count in sorted(
                term_count.items(),
                key=lambda x: (-x[1], x[0])
            )[:MAX_SYNONYMS_PER_ENTITY]
        ]

        if not top_terms:
            continue

        # Use the highest-frequency synonym as display term in the candidate list.
        display_term = top_terms[0]

        # SapBERT input text.
        # Example:
        # CCL2 MCP1 MCP-1 SCYA2 human
        entity_text = " ".join(top_terms) + " " + species

        rows.append({
            "gene_id": str(gene_id),
            "tax_id": str(tax_id),
            "species": str(species),
            "term": str(display_term),
            "entity_text": str(entity_text),
        })

    df = pd.DataFrame(rows)

    print("Final KB entity rows:", len(df))
    print(df.head().to_string(index=False))

    return df


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=64):
    """
    Encode texts with SapBERT.

    BioELQA describes SapBERT(m)[CLS], so this version uses CLS embedding
    instead of mean pooling.
    """

    embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)

        # CLS embedding
        emb = outputs.last_hidden_state[:, 0, :]

        # L2 normalize. Dot product = cosine similarity.
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        embs.append(emb.cpu().numpy().astype("float32"))

    return np.vstack(embs).astype("float32")


def parse_gold(x):
    return set(str(x).split("|"))


def main():
    print("=" * 80)
    print("Loading KB...")
    kb = load_kb(KB)

    print("=" * 80)
    print("Loading test mentions...")
    dev = pd.read_csv(MENTIONS, sep="\t")

    print("Mention file:", MENTIONS)
    print("Rows:", len(dev))
    print("Columns:", list(dev.columns))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Device:", device)

    print("Loading SapBERT:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("=" * 80)
    print("Encoding KB entities...")

    entity_texts = kb["entity_text"].astype(str).tolist()

    entity_embs = encode_texts(
        entity_texts,
        tokenizer,
        model,
        device,
        batch_size=ENCODE_BATCH_SIZE,
    )

    print("Entity embeddings shape:", entity_embs.shape)

    print("=" * 80)
    print("Encoding test mentions...")

    mention_texts = dev["mention"].astype(str).tolist()

    mention_embs = encode_texts(
        mention_texts,
        tokenizer,
        model,
        device,
        batch_size=ENCODE_BATCH_SIZE,
    )

    print("Mention embeddings shape:", mention_embs.shape)

    print("=" * 80)
    print("Retrieving TopK...")

    out_rows = []
    oracle_hit = 0
    total = 0

    retrieve_more = min(len(kb), TOPK * 5)

    for start in tqdm(range(0, len(dev), RETRIEVAL_BATCH_SIZE), desc="retrieving"):
        end = min(len(dev), start + RETRIEVAL_BATCH_SIZE)

        batch_mentions = mention_embs[start:end]

        # Batch similarity. Avoid constructing one huge full matrix.
        scores = np.matmul(batch_mentions, entity_embs.T)

        for bi in range(end - start):
            global_idx = start + bi
            r = dev.iloc[global_idx]

            score_row = scores[bi]

            if retrieve_more < len(score_row):
                top_idx = np.argpartition(-score_row, retrieve_more - 1)[:retrieve_more]
                top_idx = top_idx[np.argsort(-score_row[top_idx])]
            else:
                top_idx = np.argsort(-score_row)

            candidates = []
            seen_gid = set()

            for j in top_idx:
                item = kb.iloc[int(j)]
                gid = str(item["gene_id"])

                if gid in seen_gid:
                    continue

                seen_gid.add(gid)

                candidates.append(
                    f'{gid}::{item["tax_id"]}::{item["species"]}::{item["term"]}'
                )

                if len(candidates) >= TOPK:
                    break

            if "gold_geneid" in dev.columns:
                gold = parse_gold(r["gold_geneid"])
            elif "gold_gene_ids" in dev.columns:
                gold = parse_gold(r["gold_gene_ids"])
            else:
                gold = set()

            cand_gids = set(c.split("::")[0] for c in candidates)

            total += 1
            if gold and (gold & cand_gids):
                oracle_hit += 1

            out = r.to_dict()
            out["candidates"] = "|".join(candidates)
            out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT, sep="\t", index=False)

    print("=" * 80)
    print("Saved:", OUT)
    print("Total:", total)
    print("Oracle hit:", oracle_hit)
    print("Candidate recall@20:", oracle_hit / total if total else 0)


if __name__ == "__main__":
    main()
