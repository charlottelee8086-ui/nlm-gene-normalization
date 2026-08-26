# -*- coding: utf-8 -*-

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEV = Path("../bioelqa_dev_mentions.tsv")
KB = Path("../ncbi_symbol_synonym_taxid_kb.tsv")

OUT = Path("bioelqa_dev_candidates_sapbert_top20.tsv")

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
TOPK = 20
BATCH_SIZE = 64

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


def normalize_text(x):
    return str(x).strip()


def load_kb(path):
    """
    Expected flexible format:
    tax_id, gene_id, term, count
    or tax_id, term, gene_id
    or gene_id, tax_id, term, count

    We inspect columns by position and keep it robust.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        print("KB header:", header)

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            # Try to infer columns from header
            h = [x.lower() for x in header]

            def get_col(names, default=None):
                for name in names:
                    if name in h:
                        return parts[h.index(name)]
                return default

            tax_id = get_col(["tax_id", "taxid"], None)
            gene_id = get_col(["gene_id", "geneid"], None)
            term = get_col(["term", "symbol", "name", "alias", "synonym"], None)

            if tax_id is None or gene_id is None or term is None:
                # fallback for common formats
                # format seen in previous scripts likely: term, gene_id, tax_id, count
                # but we keep several possibilities
                if len(parts) >= 4:
                    # choose arrangement by numeric-looking fields
                    nums = [p.isdigit() for p in parts[:4]]
                    # gene_id and tax_id are numeric, term is not
                    if nums[0] and nums[1] and not nums[2]:
                        tax_id, gene_id, term = parts[0], parts[1], parts[2]
                    elif not nums[0] and nums[1] and nums[2]:
                        term, gene_id, tax_id = parts[0], parts[1], parts[2]
                    elif nums[0] and not nums[1] and nums[2]:
                        gene_id, term, tax_id = parts[0], parts[1], parts[2]
                    else:
                        continue
                else:
                    continue

            term = term.strip()
            gene_id = str(gene_id).strip()
            tax_id = str(tax_id).strip()

            if not term or not gene_id or not tax_id:
                continue

            species = SPECIES_NAME.get(tax_id, f"taxid_{tax_id}")

            # entity text for SapBERT retrieval
            entity_text = f"{term} {species}"

            rows.append({
                "gene_id": gene_id,
                "tax_id": tax_id,
                "species": species,
                "term": term,
                "entity_text": entity_text,
            })

    df = pd.DataFrame(rows).drop_duplicates(["gene_id", "tax_id", "term"])
    print("KB rows:", len(df))
    print(df.head())
    return df


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device):
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu().numpy())
    return np.vstack(embs)


def parse_gold(x):
    return set(str(x).split("|"))


def main():
    print("Loading KB...")
    kb = load_kb(KB)

    print("Loading dev...")
    dev = pd.read_csv(DEV, sep="\t")
    print("Dev rows:", len(dev))
    print("Dev columns:", list(dev.columns))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading SapBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # To reduce duplicate entities with same GeneID, keep term-level records.
    entity_texts = kb["entity_text"].astype(str).tolist()

    print("Encoding KB entities...")
    entity_embs = encode_texts(entity_texts, tokenizer, model, device)

    mention_texts = []
    for _, r in dev.iterrows():
        mention = str(r["mention"])
        # start simple: mention only, faithful to SapBERT retrieval
        mention_texts.append(mention)

    print("Encoding dev mentions...")
    mention_embs = encode_texts(mention_texts, tokenizer, model, device)

    print("Retrieving TopK...")
    # cosine similarity because vectors are normalized
    sim = np.matmul(mention_embs, entity_embs.T)

    out_rows = []
    oracle_hit = 0
    total = 0

    for idx, r in tqdm(dev.iterrows(), total=len(dev), desc="writing"):
        scores = sim[idx]
        top_idx = np.argsort(-scores)[:TOPK * 5]  # retrieve more then dedup by GeneID

        candidates = []
        seen_gid = set()

        for j in top_idx:
            item = kb.iloc[j]
            gid = str(item["gene_id"])
            if gid in seen_gid:
                continue
            seen_gid.add(gid)
            candidates.append(
                f'{gid}::{item["tax_id"]}::{item["species"]}::{item["term"]}'
            )
            if len(candidates) >= TOPK:
                break

        gold = parse_gold(r["gold_geneid"] if "gold_geneid" in dev.columns else r["gold_gene_ids"])
        cand_gids = set([c.split("::")[0] for c in candidates])

        total += 1
        if gold & cand_gids:
            oracle_hit += 1

        out = r.to_dict()
        out["candidates"] = "|".join(candidates)
        out_rows.append(out)

    pd.DataFrame(out_rows).to_csv(OUT, sep="\t", index=False)

    print("Saved:", OUT)
    print("Total:", total)
    print("Oracle hit:", oracle_hit)
    print("Candidate recall:", oracle_hit / total if total else 0)


if __name__ == "__main__":
    main()
