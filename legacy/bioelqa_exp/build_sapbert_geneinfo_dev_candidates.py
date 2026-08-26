# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEV = Path("../bioelqa_dev_mentions.tsv")
GENE_INFO = Path("../gene_info")
OUT = Path("bioelqa_dev_candidates_sapbert_geneinfo_top20.tsv")

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
TOPK = 20
BATCH_SIZE = 128
MAX_SYNONYMS = 8

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

KEEP_TAXIDS = set(SPECIES_NAME.keys())


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device):
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu().numpy())
    return np.vstack(embs)


def load_gene_info(path):
    rows = []

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="reading gene_info"):
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            tax_id = parts[0]
            gene_id = parts[1]
            symbol = parts[2]
            synonyms = parts[4]
            description = parts[8]

            # 先只保留常见物种，速度快很多，也更符合 NLM-Gene 常见数据
            if tax_id not in KEEP_TAXIDS:
                continue

            species = SPECIES_NAME.get(tax_id, "taxid_" + tax_id)

            terms = []
            if symbol and symbol != "-":
                terms.append(symbol)

            if synonyms and synonyms != "-":
                for s in synonyms.split("|"):
                    s = s.strip()
                    if s and s != "-":
                        terms.append(s)

            # 去重并限制 synonym 数量
            clean_terms = []
            seen = set()
            for t in terms:
                key = t.lower()
                if key in seen:
                    continue
                seen.add(key)
                clean_terms.append(t)
                if len(clean_terms) >= MAX_SYNONYMS:
                    break

            if not clean_terms:
                continue

            # 每个 GeneID + TaxID 只生成一个 entity text
            entity_text = " ".join(clean_terms + [species, description])

            rows.append({
                "gene_id": gene_id,
                "tax_id": tax_id,
                "species": species,
                "term": clean_terms[0],
                "entity_text": entity_text,
            })

    df = pd.DataFrame(rows).drop_duplicates(["gene_id", "tax_id"])
    print("Gene-level entities:", len(df))
    print(df.head())
    return df


def parse_gold(x):
    return set(str(x).split("|"))


def main():
    print("Loading gene_info entities...")
    ent_df = load_gene_info(GENE_INFO)

    print("Loading dev mentions...")
    dev = pd.read_csv(DEV, sep="\t")
    print("Dev rows:", len(dev))
    print("Dev columns:", list(dev.columns))

    gold_col = "gold_geneid" if "gold_geneid" in dev.columns else "gold_gene_ids"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading SapBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("Encoding gene-level entities...")
    entity_embs = encode_texts(ent_df["entity_text"].astype(str).tolist(), tokenizer, model, device)

    print("Encoding dev mentions...")
    mention_embs = encode_texts(dev["mention"].astype(str).tolist(), tokenizer, model, device)

    print("Retrieving candidates...")
    sim = np.matmul(mention_embs, entity_embs.T)

    rows = []
    hit = 0
    total = 0

    for i, r in tqdm(dev.iterrows(), total=len(dev), desc="writing"):
        scores = sim[i]
        top_idx = np.argsort(-scores)[:TOPK]

        candidates = []
        for j in top_idx:
            item = ent_df.iloc[j]
            candidates.append(
                f'{item["gene_id"]}::{item["tax_id"]}::{item["species"]}::{item["term"]}'
            )

        gold = parse_gold(r[gold_col])
        cand_gids = set(c.split("::")[0] for c in candidates)

        total += 1
        if gold & cand_gids:
            hit += 1

        out = r.to_dict()
        out["candidates"] = "|".join(candidates)
        rows.append(out)

    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)

    print("Saved:", OUT)
    print("Total:", total)
    print("Oracle hit:", hit)
    print("Candidate recall:", hit / total if total else 0)


if __name__ == "__main__":
    main()
